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
