"""A CrewAI crew that stores findings in Synap Core and recalls them across runs.

    pip install crewai langchain-openai maximem-synap
    unset SYNAP_API_KEY                  # ensures SDK acts against Core, not Cloud
    export SYNAP_BASE_URL=http://127.0.0.1:8000
    export OPENAI_API_KEY=sk-...
    python crewai-memory.py

Synap Core runs locally with no API key. On a business (strict) scoping install set
SYNAP_CUSTOMER_ID=acme so every call carries a customer id; on a consumer install leave it unset
(a customer id is refused there).
"""
from __future__ import annotations

import asyncio
import os

from crewai import Agent, Crew, Process, Task
from maximem_synap import MaximemSynapSDK

# Prevent the SDK from falling back to Cloud mode if an API key was left in the environment
os.environ.pop("SYNAP_API_KEY", None)

CUSTOMER = {"customer_id": os.environ["SYNAP_CUSTOMER_ID"]} if os.environ.get("SYNAP_CUSTOMER_ID") else {}


def format_context(ctx) -> str:
    """Format facts and preferences retrieved from Synap Core."""
    entries = []
    for section in ("facts", "preferences"):
        for item in getattr(ctx, section, []) or []:
            entries.append(getattr(item, "content", str(item)))
    return "\n".join(f"- {e}" for e in entries) if entries else "No past memory found."


async def run_first_session(sdk: MaximemSynapSDK, user_id: str) -> str:
    """Run 1: Crew learns user requirements and stores them into Synap Core."""
    analyst = Agent(
        role="Client Intake Specialist",
        goal="Extract and summarize client requirements and technical preferences.",
        backstory="You interview users and organize their key preferences into concise bullet points.",
        verbose=False,
    )

    intake_task = Task(
        description=(
            "The client says: 'We are building a healthcare app in Python and PostgreSQL. "
            "We strictly use HIPAA compliance guidelines and prefer asynchronous architecture.' "
            "Summarize the key architectural facts and compliance preferences."
        ),
        expected_output="A concise summary of client requirements and architecture choices.",
        agent=analyst,
    )

    crew = Crew(agents=[analyst], tasks=[intake_task], process=Process.sequential)
    result = crew.kickoff()
    summary_text = str(result)

    # Store findings in Synap Core so future runs can remember them
    doc = f"User requirements and preferences from initial intake:\n{summary_text}"
    created = await sdk.memories.create(
        document=doc,
        document_type="document",
        user_id=user_id,
        **CUSTOMER,
    )
    await sdk.memories.wait_for_completion(created.ingestion_id)
    return summary_text


async def run_second_session(sdk: MaximemSynapSDK, user_id: str) -> str:
    """Run 2: Crew recalls past memories from Synap Core and uses them in planning."""
    # Pull remembered context for this user from Synap Core
    context = await sdk.user.context.fetch(
        user_id=user_id,
        search_query=["architecture requirements", "compliance guidelines"],
        **CUSTOMER,
    )
    recalled_notes = format_context(context)

    architect = Agent(
        role="Senior Solutions Architect",
        goal="Design architecture plans strictly respecting recalled client preferences.",
        backstory=f"You are a systems architect. You have these verified past client memories:\n{recalled_notes}",
        verbose=False,
    )

    planning_task = Task(
        description=(
            "Recommend three specific Python libraries and backend patterns tailored to the "
            "client's recalled preferences (e.g. database async drivers and security tooling)."
        ),
        expected_output="A recommendations list tailored to the client's recalled context.",
        agent=architect,
    )

    crew = Crew(agents=[architect], tasks=[planning_task], process=Process.sequential)
    result = crew.kickoff()
    return str(result)


async def main() -> None:
    user = "client_enterprise_alpha"
    sdk = MaximemSynapSDK()
    await sdk.initialize()

    try:
        print("=== Run 1: Crew gathers and persists client memory to Synap Core ===")
        intake_summary = await run_first_session(sdk, user)
        print(f"Summary stored:\n{intake_summary}\n")

        print("=== Run 2: Crew reads memory back from Synap Core on subsequent run ===")
        recommendations = await run_second_session(sdk, user)
        print(f"Personalized Plan based on recalled memory:\n{recommendations}")

    finally:
        await sdk.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
