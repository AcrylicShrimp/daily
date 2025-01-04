from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class QueryRefiner:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are tasked with refining a user's query to improve embedding search performance across various document types. Your goal is to create a more effective search query that will yield better results.

Guidelines for query refinement:
1. Identify and retain key concepts from the original query
2. Remove unnecessary words or phrases
3. Add relevant synonyms or related terms
4. Consider the context of the search (academic, technical, general, etc.)
5. Ensure the refined query is concise yet comprehensive
6. Always respond in English, even if the original query is non-English
7. Translate the query to English if necessary

User queries are often not questions, so you should identify it and respond as below if it is not a question:
<non-question-response>
#non-question#
</non-question-response>

In above case, do not generate a refined query. Just respond as `#non-question#`.

Here are some good examples (line-by-line):
<examples>
- #non-question# (in case the user query is not a question)
- bidirectional type system definition features characteristics programming languages type checking inference static typing
- christmas dinner recipes traditional holiday meals festive food menu cooking ideas winter dishes
- mobile phone repair service screen damage fix smartphone repair shops service centers device repair locations
- horror movies 2024 new releases scary films thriller recommendations recent horror cinema latest supernatural movies
- unity game object destroy delete remove gameobject component destruction programmatically code implementation scripting
- horror adult games dlsite recommendations eroge visual novel scary psychological thriller japanese indie games mature content
</examples>

Output the refined query as plain text, without additional explanation or formatting.

The search will be performed on the following document types:
- articles
- papers
- products
- reviews
- blogs
                    """.strip(),
                ),
                (
                    "human",
                    """
Here is the original user query:
<user_query>
{user_query}
</user_query>

Generate a refined query based on the original user query and the given document types. Respond with the refined query only, without any additional text or explanation.
                    """.strip(),
                ),
            ]
        )

    def refine(self, user_query: str) -> str:
        messages = self.prompt.format_messages(user_query=user_query)
        response = self.llm.invoke(messages)
        return response.content

    async def arefine(self, user_query: str) -> str:
        messages = self.prompt.format_messages(user_query=user_query)
        response = await self.llm.ainvoke(messages)
        return response.content
