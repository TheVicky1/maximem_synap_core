"""Run Synap Core completely locally with Ollama and zero API keys.

1. Pull a model with Ollama:
       ollama pull llama3.2          # 3B parameters; fast and capable
       # Alternatively: ollama pull qwen2.5:7b or ollama pull mistral

2. Set up .env in the repository root:
       SYNAP_LLM_PROVIDER=ollama
       SYNAP_LLM_MODEL=llama3.2
       OLLAMA_NUM_CTX=16384
       SYNAP_LLM_READ_TIMEOUT_SECONDS=300

   On macOS / Windows (Docker Desktop):
       OLLAMA_BASE_URL=http://host.docker.internal:11434

   On Linux:
       OLLAMA_BASE_URL=http://172.17.0.1:11434
       (Ollama listens on 127.0.0.1 by default, which Docker containers cannot reach.
       Set OLLAMA_HOST=172.17.0.1:11434 before starting Ollama to match the docker0 bridge;
       see skills/synap-core/reference/model-setup.md).

3. Start Synap Core:
       docker compose -f deploy/docker-compose.yml up -d

4. Run this script:
       pip install maximem-synap
       unset SYNAP_API_KEY          # ensures SDK acts against Core, not Cloud
       export SYNAP_BASE_URL=http://127.0.0.1:8000
       python skills/synap-core/examples/ollama-local.py

   If the server has a token, also export SYNAP_API_TOKEN. On a business (strict) install set
   SYNAP_CUSTOMER_ID=acme so every call carries a customer id; on a consumer install leave it unset
   (a customer id is refused there).

Note:
- Local models take longer per chunk than hosted APIs; SYNAP_LLM_READ_TIMEOUT_SECONDS is raised to 300 in .env to prevent client timeouts during extraction.
"""
from __future__ import annotations

import asyncio
import os
import sys

from maximem_synap import MaximemSynapSDK

# Prevent the SDK from falling back to Cloud mode if an API key was left in the environment
os.environ.pop("SYNAP_API_KEY", None)

CUSTOMER = {"customer_id": os.environ["SYNAP_CUSTOMER_ID"]} if os.environ.get("SYNAP_CUSTOMER_ID") else {}


async def main() -> None:
    sdk = MaximemSynapSDK()
    await sdk.initialize()
    user_id = "local_dev_user"

    try:
        print("[1/3] Writing conversation to local Synap Core (Ollama extraction)...")
        document = (
            "User: I live in Amsterdam and my favorite drink is iced matcha latte.\n"
            "Assistant: Great, I will keep that in mind!"
        )
        result = await sdk.memories.create(
            document=document,
            document_type="ai-chat-conversation",
            user_id=user_id,
            **CUSTOMER,
        )
        print(f"      Ingestion job started (ID: {result.ingestion_id})")

        print("[2/3] Waiting for Ollama model to process and extract memories...")
        status = await sdk.memories.wait_for_completion(result.ingestion_id, timeout_seconds=400)
        print(f"      Ingestion finished: {status.memories_created} memories created (status: {status.status.value})")

        print("[3/3] Recalling memories using search queries...")
        context = await sdk.user.context.fetch(
            user_id=user_id,
            search_query=["city location", "favorite beverage"],
            **CUSTOMER,
        )

        all_memories: list[str] = []
        for section in ("facts", "preferences", "episodes"):
            for item in getattr(context, section, []) or []:
                all_memories.append(f"[{section.upper()}] {getattr(item, 'content', str(item))}")

        if all_memories:
            print("\nRecalled Memories:")
            for m in all_memories:
                print(f"  - {m}")
        else:
            print("No memories recalled.")

    finally:
        await sdk.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
