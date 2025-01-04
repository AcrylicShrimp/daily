from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

from document_storage import DocumentStorage

search = DuckDuckGoSearchResults(num_results=10, output_format="json")
document_storage = DocumentStorage()


@tool
async def search_web(query: str) -> dict[str, any]:
    """
    Search the web for new information. It uses DuckDuckGo Search.

    Args:
        query: The query to search for.

    Returns:
        The documents that are relevant to the given query, from the web.

    Note:
        When providing a query, do not pass the original query. Instead, generate a refined query, following these guidelines:

        1. Utilize the search engine's advanced search features to improve the search results
        2. Identify and retain key concepts from the original query
        3. Remove unnecessary words or phrases
        4. Add relevant synonyms or related terms
        5. Consider the context of the search (academic, technical, general, etc.)
        6. Ensure the refined query is concise yet comprehensive
        7. Always respond in English, even if the original query is non-English
        8. Translate the query to English if necessary
    """

    return {
        "query": query,
        "result": await search.ainvoke(query),
    }


@tool
async def search_documents(query: str) -> dict[str, any]:
    """
    Search for indexed documents that are relevant to the given query.

    The search will be performed on the following document types:
        - articles
        - papers
        - products
        - reviews
        - blogs

    Args:
        query: The query to search for.

    Returns:
        The documents that are relevant to the given query.

    Note:
        When providing a query, do not pass the original query. Instead, generate a refined query, following these guidelines:

        1. Identify and retain key concepts from the original query
        2. Remove unnecessary words or phrases
        3. Add relevant synonyms or related terms
        4. Consider the context of the search (academic, technical, general, etc.)
        5. Ensure the refined query is concise yet comprehensive
        6. Always respond in English, even if the original query is non-English
        7. Translate the query to English if necessary

        Here are some good examples (line-by-line):
        <examples>
        - bidirectional type system definition features characteristics programming languages type checking inference static typing
        - christmas dinner recipes traditional holiday meals festive food menu cooking ideas winter dishes
        - mobile phone repair service screen damage fix smartphone repair shops service centers device repair locations
        - horror movies 2024 new releases scary films thriller recommendations recent horror cinema latest supernatural movies
        - unity game object destroy delete remove gameobject component destruction programmatically code implementation scripting
        - horror adult games dlsite recommendations eroge visual novel scary psychological thriller japanese indie games mature content
        </examples>
    """
    docs = await document_storage.aquery(query)

    if len(docs) == 0:
        return {
            "query": query,
            "documents": [],
            "warning": "no relevant documents found",
        }

    return {
        "query": query,
        "documents": [doc.page_content.strip() for doc in docs],
    }
