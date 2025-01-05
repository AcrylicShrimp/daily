import json
from typing import Generator, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    BaseMessageChunk,
    AIMessage,
    HumanMessage,
    ToolMessage,
    ToolCall,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from llms.agent_detail.prompts import SYSTEM_PROMPT, QUERY_PROMPT, CONTINUE_PROMPT
from llms.agent_detail.tools import search


class Agent:
    def __init__(self, language: str = "en"):
        self.language = language
        self.history = []
        self.llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.4)
        self.llm = self.llm.bind_tools(
            [
                search,
            ]
        )
        self.query_prompt = ChatPromptTemplate.from_messages(
            [
                SYSTEM_PROMPT,
                MessagesPlaceholder(variable_name="history"),
                QUERY_PROMPT,
            ]
        )
        self.continue_prompt = ChatPromptTemplate.from_messages(
            [
                SYSTEM_PROMPT,
                MessagesPlaceholder(variable_name="history"),
                CONTINUE_PROMPT,
            ]
        )

    def run(
        self,
        query: str,
    ) -> "AgentIterator":
        return AgentIterator(self, query, self.language)

    async def handle_tool_calls(self, tool_calls: list[ToolCall]) -> "AgentIterator":
        self.history.append(AIMessage(content="", tool_calls=tool_calls))

        for tool_call in tool_calls:
            tools = {
                "search": search,
            }
            tool = tools.get(tool_call["name"])

            if tool is None:
                result = {
                    "error": f"invalid tool name {tool_call['name']}",
                }
            else:
                result = await tool.ainvoke(tool_call["args"])

            self.history.append(
                ToolMessage(
                    tool_call_id=tool_call["id"],
                    content=json.dumps(result),
                    status="error" if "error" in result else "success",
                )
            )

        return AgentIterator(self, "", self.language)


class AgentIterator:
    def __init__(self, agent: Agent, query: str, language: str):
        self.agent = agent
        self.query = query
        self.language = language

    def __iter__(self) -> Generator[tuple[bool, str, list[ToolCall]], None, None]:
        prompt_set = (
            self.agent.continue_prompt
            if len(self.query) == 0
            else self.agent.query_prompt
        )
        prompt = prompt_set.format_messages(
            history=self.agent.history,
            query=self.query,
            language=self.language,
        )
        collector = AgentResponseChunkCollector()

        for chunk in self.agent.llm.stream(prompt):
            partial_answer = collector.handle_chunk(chunk)
            yield False, partial_answer, []

        answer, tool_calls = collector.finalize()

        self.agent.history.append(HumanMessage(content=self.query))
        self.agent.history.append(AIMessage(content=answer))

        yield True, answer, tool_calls


class PartialToolCall(TypedDict):
    id: str
    name: str
    args: str


class AgentResponseChunkCollector:
    def __init__(self):
        self.answer = ""
        self.tool_calls: dict[int, PartialToolCall] = {}

    def handle_chunk(self, chunk: BaseMessageChunk) -> str:
        content = chunk.content

        if isinstance(content, str):
            self.answer += content
            return content

        partial_answer = ""

        for elem in content:
            if isinstance(elem, str):
                partial_answer += elem
                continue

            if elem["type"] == "text":
                partial_answer += elem["text"]
                continue

            if elem["type"] == "tool_use":
                if "name" in elem and "id" in elem and "index" in elem:
                    self.tool_calls[elem["index"]] = {
                        "id": elem["id"],
                        "name": elem["name"],
                        "args": "",
                    }
                elif (
                    "index" in elem
                    and "partial_json" in elem
                    and elem["index"] in self.tool_calls
                ):
                    self.tool_calls[elem["index"]]["args"] += elem["partial_json"]

        self.answer += partial_answer
        return partial_answer

    def finalize(self) -> tuple[str, list[ToolCall]]:
        tool_calls = list(self.tool_calls.items())
        tool_calls.sort(key=lambda x: x[0])
        return self.answer, [
            {
                "id": x[1]["id"],
                "name": x[1]["name"],
                "args": json.loads(x[1]["args"]),
            }
            for x in tool_calls
        ]
