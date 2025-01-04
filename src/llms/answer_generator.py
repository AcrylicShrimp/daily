from datetime import datetime
from typing import Iterator

from langchain_core.documents import Document
from langchain_core.messages import BaseMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class AnswerGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an AI assistant tasked with answering user queries based on provided information. Your goal is to generate accurate, helpful, and kind responses. Follow these instructions carefully:

1. You will be given some related documents retrieved by a document searching system. These documents contain information relevant to the user's query.

2. Analyze the related documents and the user query. Look for information in the documents that directly addresses the user's question or is closely related to the topic of the query.

3. Formulate your response based on the following guidelines:
   a. If the related documents contain information that answers the user's query, use that information as the basis for your response.
   b. If the documents provide partial information, use what is available and acknowledge any gaps in the information.
   c. If the documents do not contain relevant information to answer the query, politely state that you don't have enough information to provide a complete answer.

4. When crafting your response:
   a. Be kind, soft, and empathetic in your tone.
   b. Use clear and simple language.
   c. If appropriate, offer additional context or explanations to help the user better understand the answer.
   d. If the query is unclear, politely ask for clarification.

5. Provide your answer as plain text. Your response should be structured as follows:
   - A gentle greeting or acknowledgment of the user's query
   - The main body of your answer, based on the information from the related documents
   - Any necessary caveats or limitations of the information provided
   - A polite closing statement

Remember, your primary goal is to be helpful and kind while providing accurate information based on the given documents. Do not invent information or make assumptions beyond what is provided in the related documents.
                    """.strip(),
                ),
                (
                    "human",
                    """
Here are the related documents:

<related_documents>
{related_documents}
</related_documents>

The user has submitted the following query:

<user_query>
{user_query}
</user_query>

Now, please generate your response to the user query.
                    """.strip(),
                ),
            ]
        )

    def generate(
        self, related_documents: list[Document], user_query: str
    ) -> Iterator[BaseMessageChunk]:
        docs = stringify_documents(related_documents)
        messages = self.prompt.format_messages(
            related_documents=docs,
            user_query=user_query,
        )
        return self.llm.stream(messages)


def current_time() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"""
Here is the current date and time, refer it in your response if necessary:
<current_time>
{now}
</current_time>
                """.strip()
    return formatted


def stringify_documents(documents: list[Document]) -> str:
    return "\n\n".join(
        [current_time(), *[doc.page_content.strip() for doc in documents]]
    )
