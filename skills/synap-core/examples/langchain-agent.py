"""A LangChain agent that stores user conversations in Synap Core and retrieves context.

    pip install langchain langchain-openai maximem-synap
    export SYNAP_BASE_URL=http://127.0.0.1:8000
    export OPENAI_API_KEY=sk-...
    python langchain-agent.py

Synap Core runs locally with no API key. On a business (strict) install set
SYNAP_CUSTOMER_ID=acme so every call carries a customer id; on a consumer install leave it unset
(a customer id is refused there).
"""
from __future__ import annotations

import asyncio
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from maximem_synap import MaximemSynapSDK

CUSTOMER = {"customer_id": os.environ["SYNAP_CUSTOMER_ID"]} if os.environ.get("SYNAP_CUSTOMER_ID") else {}


def format_context(ctx) -> str:
    """Extract facts and preferences from the Synap context response."""
    items = []
    for section in ("facts", "preferences"):
        for item in getattr(ctx, section, []) or []:
            items.append(getattr(item, "content", str(item)))
    return "\n".join(f"- {i}" for i in items) if items else "No previous memories found."


async def run_agent(sdk: MaximemSynapSDK, llm: ChatOpenAI, user_id: str, user_message: str) -> str:
    # 1. Fetch relevant memories from Synap Core as context before answering
    context = await sdk.user.context.fetch(user_id=user_id, search_query=[user_message], **CUSTOMER)
    memory_notes = format_context(context)

    # 2. Invoke the LangChain model with user input and retrieved memory context
    messages = [
        SystemMessage(content=f"You are a helpful assistant. Known user facts and preferences:\n{memory_notes}"),
        HumanMessage(content=user_message),
    ]
    response = await llm.ainvoke(messages)
    assistant_reply = str(response.content)

    # 3. Store conversation turn in Synap Core so subsequent runs remember it
    turn = f"User: {user_message}\nAssistant: {assistant_reply}"
    created = await sdk.memories.create(
        document=turn,
        document_type="ai-chat-conversation",
        user_id=user_id,
        **CUSTOMER,
    )
    await sdk.memories.wait_for_completion(created.ingestion_id)
    return assistant_reply


async def main() -> None:
    user = "alice_wonder"
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

    sdk = MaximemSynapSDK()
    await sdk.initialize()
    try:
        print("--- Turn 1: Sharing preferences ---")
        reply1 = await run_agent(sdk, llm, user, "I prefer oat milk in my coffee and I am vegetarian.")
        print(f"Assistant: {reply1}\n")

        print("--- Turn 2: Asking for a recommendation using recalled context ---")
        reply2 = await run_agent(sdk, llm, user, "Can you suggest a quick breakfast for me?")
        print(f"Assistant: {reply2}")
    finally:
        await sdk.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
