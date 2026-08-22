"""Run a minimal OpenAI Agents SDK agent through OpenRouter's Ox Alpha."""

import os
import sys

from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from dotenv import load_dotenv
from openai import AsyncOpenAI


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("openrouter_api")
    if not api_key:
        raise SystemExit(
            "Missing OpenRouter key. Set OPENROUTER_API_KEY (or openrouter_api) "
            "in the environment or .env file."
        )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    model = OpenAIChatCompletionsModel(
        model="stealth/ox-alpha",
        openai_client=client,
    )

    # OpenAI tracing requires a separate OpenAI Platform API key.
    set_tracing_disabled(True)

    agent = Agent(
        name="Ox Assistant",
        instructions="You are a concise, helpful assistant.",
        model=model,
    )
    prompt = " ".join(sys.argv[1:]) or "Explain anomaly detection in one sentence."
    result = Runner.run_sync(agent, prompt)
    print(result.final_output)


if __name__ == "__main__":
    main()
