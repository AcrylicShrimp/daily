from langchain_core.tools import tool

from document_storage import DocumentStorage

document_storage = DocumentStorage()


@tool
async def search_documents(query: str) -> list[str]:
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
    return [doc.page_content.strip() for doc in docs]
