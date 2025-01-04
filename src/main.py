#! /usr/bin/env python3

import asyncio

from langchain_core.messages import ToolCall

from llms.agent import Agent


async def main():
    language = input("Language: ").strip()
    agent = Agent(language)

    while True:
        query = input("User: ").strip()

        if len(query) == 0:
            break

        tool_calls_after_answer: list[ToolCall] = []

        while True:
            print()
            print("Assistant: ", end="")

            if len(tool_calls_after_answer) != 0:
                print("Executing tools:")

                for tool_call in tool_calls_after_answer:
                    print(f"- Tool: {tool_call['name']} ({tool_call['args']})")

                print()

            stream = (
                agent.run(query)
                if len(tool_calls_after_answer) == 0
                else await agent.handle_tool_calls(tool_calls_after_answer)
            )

            for is_done, partial_answer, tool_calls in stream:
                if is_done:
                    tool_calls_after_answer = tool_calls
                    break

                print(partial_answer, end="", flush=True)

            print()

            if len(tool_calls_after_answer) == 0:
                break

        print()


if __name__ == "__main__":
    asyncio.run(main())
