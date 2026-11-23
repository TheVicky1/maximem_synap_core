"""Tests for cache invalidation across scopes (#35).

When memories are written or updated at client-scope (customer_id empty or 'global'),
all query results under that client (including user and customer scopes) must be invalidated
so that subsequent fetches immediately reflect the new memory rather than serving stale cache.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from synap.cloud.application.client_api.retrieval_manager import RetrievalManager


def test_client_scope_invalidation_clears_all_client_queries():
    async def _test():
        manager = RetrievalManager.__new__(RetrievalManager)
        manager._adjacency_l2 = MagicMock()
        manager._adjacency_l2.delete_scope.return_value = 1
        manager.cache = AsyncMock()
        manager.cache.scan = AsyncMock(
            side_effect=[
                (42, ["query:cli_123:cust_a:hash1", "query:cli_123:cust_b:hash2"]),
                (0, ["query:cli_123:global:hash3"]),
            ]
        )
        manager.cache.delete = AsyncMock(return_value=3)

        deleted = await manager.invalidate_user_cache(
            instance_id="inst_1",
            client_id="cli_123",
            customer_id="",
        )

        assert deleted == 3
        # Scan should use pattern matching all queries under the client
        assert manager.cache.scan.call_count == 2
        manager.cache.scan.assert_any_call(cursor=0, match="query:cli_123:*", count=100)
        manager.cache.scan.assert_any_call(cursor=42, match="query:cli_123:*", count=100)
        # Adjacency should drop all variants under the client
        manager._adjacency_l2.delete_scope.assert_called_once_with("cli_123")

    asyncio.run(_test())


def test_customer_scope_invalidation_targets_only_that_customer():
    async def _test():
        manager = RetrievalManager.__new__(RetrievalManager)
        manager._adjacency_l2 = MagicMock()
        manager._adjacency_l2.delete_scope.return_value = 1
        manager.cache = AsyncMock()
        manager.cache.scan = AsyncMock(
            return_value=(0, ["query:cli_123:cust_a:hash1"])
        )
        manager.cache.delete = AsyncMock(return_value=1)

        deleted = await manager.invalidate_user_cache(
            instance_id="inst_1",
            client_id="cli_123",
            customer_id="cust_a",
            user_id="user_9",
        )

        assert deleted == 1
        # Scan should target only the specific customer
        manager.cache.scan.assert_called_once_with(cursor=0, match="query:cli_123:cust_a:*", count=100)
        manager.cache.delete.assert_called_once_with("query:cli_123:cust_a:hash1")
        # Adjacency drops scope key for that user under customer
        manager._adjacency_l2.delete_scope.assert_called_once_with("cli_123:cust_a:user_9")

    asyncio.run(_test())


def test_client_scope_with_global_customer_id_clears_all_client_queries():
    async def _test():
        manager = RetrievalManager.__new__(RetrievalManager)
        manager._adjacency_l2 = MagicMock()
        manager._adjacency_l2.delete_scope.return_value = 1
        manager.cache = AsyncMock()
        manager.cache.scan = AsyncMock(
            return_value=(0, ["query:cli_456:cust_x:hash1"])
        )
        manager.cache.delete = AsyncMock(return_value=1)

        deleted = await manager.invalidate_user_cache(
            instance_id="inst_2",
            client_id="cli_456",
            customer_id="global",
        )

        assert deleted == 1
        manager.cache.scan.assert_called_once_with(cursor=0, match="query:cli_456:*", count=100)
        manager._adjacency_l2.delete_scope.assert_called_once_with("cli_456")

    asyncio.run(_test())


def test_separate_writer_and_reader_managers_invalidate_client_graphs():
    async def _test():
        from synap.cloud.application.client_api.retrieval_manager import (
            RetrievalManager,
            _AdjacencyL2Cache,
            _l2_dirty_keys,
        )
        from synap.cloud.shared.scope import ScopeHierarchy

        # Clean slate for test
        _l2_dirty_keys.clear()

        # Reader manager has existing in-memory cached graphs and in-flight build
        reader = RetrievalManager.__new__(RetrievalManager)
        reader._adjacency_l2 = _AdjacencyL2Cache()
        reader._adjacency_inflight = {}
        reader.cache = None
        reader._scope_resolver = MagicMock()

        # Seed cached graphs for two different customers under cli_777
        reader._adjacency_l2.put("cli_777:cust_a:user_1", {"nodes": [1]})
        reader._adjacency_l2.put("cli_777:cust_b:user_2", {"nodes": [2]})
        reader._adjacency_inflight["cli_777:cust_a:user_1"] = "inflight_future"

        assert reader._adjacency_l2.get("cli_777:cust_a:user_1") is not None
        assert reader._adjacency_l2.get("cli_777:cust_b:user_2") is not None

        # Throwaway writer manager in another part of the system (e.g. stage4 worker)
        writer = RetrievalManager.__new__(RetrievalManager)
        writer._adjacency_l2 = _AdjacencyL2Cache()
        writer.cache = AsyncMock()
        writer.cache.scan = AsyncMock(return_value=(0, []))
        writer.cache.delete = AsyncMock(return_value=0)

        # Ingest a client-level memory (empty customer_id)
        await writer.invalidate_user_cache(
            instance_id="inst_777",
            client_id="cli_777",
            customer_id="",
        )

        # Writer set the client dirty flag in the shared module set
        assert "cli_777" in _l2_dirty_keys

        # When reader gets a request for cust_a under cli_777, it executes _get_adjacency_matrix
        reader._load_or_build_adjacency = AsyncMock(return_value={"nodes": ["fresh"]})
        scope = ScopeHierarchy(
            client_id="cli_777",
            customer_id="cust_a",
            user_id="user_1",
            instance_id="inst_777",
        )

        matrix = await reader._get_adjacency_matrix(scope)
        assert matrix == {"nodes": ["fresh"]}

        # All client-wide cached graphs in reader are dropped!
        assert reader._adjacency_l2.get("cli_777:cust_a:user_1") is None       # stale entry was purged
        assert reader._adjacency_l2.get("cli_777:cust_b:user_2") is None       # stale entry was purged
        assert "cli_777:cust_a:user_1" not in reader._adjacency_inflight
        assert "cli_777" not in _l2_dirty_keys

    asyncio.run(_test())
