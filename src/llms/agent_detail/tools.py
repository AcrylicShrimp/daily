import asyncio
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_storage import DocumentStorage
from llms.agent_detail.html_extractor import extract_html

web_search = DuckDuckGoSearchResults(num_results=10, output_format="list")
document_storage = DocumentStorage(top_k_search=40, top_k_rerank=10)
document_splitter = RecursiveCharacterTextSplitter(chunk_size=255, chunk_overlap=16)


@tool
async def search(query: str, ignore_cache: bool = False) -> any:
    """
    Search for information relevant to the given query.
    Use this tool to search basis, documents, and web to answer questions of the user.

    It will first search the cached documents. If there is no relevant information found, it will search the web and index the results before returning.

    Args:
        query: The query to search for.
        ignore_cache: Whether to ignore the cache. If `True`, it will search the web always. Defaults to `False` (use cached documents first).

    Returns:
        The information that is relevant to the given query.

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

        - bidirectional type system definition features characteristics programming languages type checking inference static typing
        - christmas dinner recipes traditional holiday meals festive food menu cooking ideas winter dishes
        - mobile phone repair service screen damage fix smartphone repair shops service centers device repair locations
        - horror movies 2024 new releases scary films thriller recommendations recent horror cinema latest supernatural movies
        - unity game object destroy delete remove gameobject component destruction programmatically code implementation scripting
        - horror adult games dlsite recommendations eroge visual novel scary psychological thriller japanese indie games mature content
        - common cold symptoms flu symptoms cold vs flu natural remedies cold treatment flu treatment viral infections respiratory illnesses
        - iPhone vs Samsung Galaxy comparison smartphone specs comparison mobile phone features comparison iPhone Samsung camera battery display
        - cover letter writing tips no experience cover letter template entry-level cover letter job application cover letter writing advice how to write a cover letter

        Do not use this tool multiple times with the same query in short time period, as it will be blocked by the server and/or just returns the same results.
    """
    try:

        async def search_cached_documents(query: str) -> list[Document]:
            return [] if ignore_cache else await document_storage.aquery(query)

        async def search_web(query: str) -> list[dict]:
            results = await web_search.ainvoke(query)
            return [
                {
                    "title": result["title"],
                    "snippet": result["snippet"],
                    "url": result["link"],
                }
                for result in results
            ]

        def serialize_documents(docs: list[Document]) -> dict:
            return [
                {
                    "url": doc.metadata["url"],
                    "title": doc.metadata["title"],
                    "timestamp": doc.metadata["timestamp"],
                    "content": doc.page_content,
                }
                for doc in docs
            ]

        documents, web_results = await asyncio.gather(
            search_cached_documents(query),
            search_web(query),
        )

        if 0 < len(documents):
            return {
                "query": query,
                "origin": "cached-documents",
                "documents": serialize_documents(documents),
            }

        if len(web_results) == 0:
            return {
                "query": query,
                "origin": "web",
                "documents": [],
                "warning": "no relevant information found",
            }

        now = datetime.now().isoformat()

        async def process_web_result(result: dict):
            try:
                content = await extract_html(result["url"])

                if content == "":
                    return

                document = Document(
                    page_content=content,
                    metadata={
                        "query": query,
                        "title": result["title"],
                        "snippet": result["snippet"],
                        "url": result["url"],
                        "timestamp": now,
                    },
                )
                chunks = document_splitter.split_documents([document])
                await document_storage.aadd_documents(chunks)
            except:
                pass

        await asyncio.gather(*[process_web_result(result) for result in web_results])

        documents = await document_storage.aquery(query)

        if 0 < len(documents):
            return {
                "query": query,
                "origin": "web",
                "documents": serialize_documents(documents),
            }

        return {
            "query": query,
            "origin": "web",
            "documents": [],
            "warning": "no relevant information found",
        }

    except Exception as e:
        return {
            "status": "error",
            "cause": str(e),
        }
