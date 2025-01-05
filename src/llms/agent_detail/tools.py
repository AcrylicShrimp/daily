import asyncio
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_storage import DocumentStorage
from llms.agent_detail.html_extractor import HtmlExtractor

search = DuckDuckGoSearchResults(num_results=10, output_format="list")
document_storage = DocumentStorage()
document_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


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

    results = await search.ainvoke(query)
    results = [
        {"title": result["title"], "snippet": result["snippet"], "url": result["link"]}
        for result in results
    ]

    async def process_result(result: dict):
        try:
            html_extractor = HtmlExtractor(result["url"])
            await html_extractor.fetch()
            result["content"] = html_extractor.extract()
        except Exception as e:
            print(f"[search_web] warning: failed to fetch url `{result['url']}`: {e}")
            return None

    results = await asyncio.gather(*[process_result(result) for result in results])
    results = [r for r in results if r is not None]

    return {
        "query": query,
        "results": results,
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
        "documents": [
            {"metadata": doc.metadata, "content": doc.page_content.strip()}
            for doc in docs
        ],
    }


@tool
async def index_documents(urls: list[str]) -> list[str]:
    """
    Index the given HTML pages.

    It will extract the content of the HTML pages and store them in the document storage.

    Args:
        urls: The URLs of the HTML pages to index.

    Returns:
        The URLs of the HTML pages that were successfully indexed.

    Note:
        It does not index raw HTML pages. Instead, it extracts the content of the HTML pages, using text density to figure out the most relevant content.
    """

    async def process_url(url: str):
        try:
            html_extractor = HtmlExtractor(url)
            await html_extractor.fetch()
            content = html_extractor.extract()
            return url, content
        except:
            return None, None

    processed = await asyncio.gather(*[process_url(url) for url in urls])
    processed = [p for p in processed if p[0] is not None]

    chunks = [
        document_splitter.split_documents(
            [Document(page_content=content, metadata={"url": url})]
        )
        for url, content in processed
    ]
    await document_storage.aadd_documents(chunks)

    return [url for url, _ in processed]
