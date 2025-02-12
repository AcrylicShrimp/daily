import asyncio
from datetime import datetime
import json

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_storage import DocumentStorage
from llms.agent_detail.extract_html.html_extractor import extract_html

web_search = DuckDuckGoSearchResults(num_results=20, output_format="list")
document_storage = DocumentStorage()
document_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
llm = ChatOpenAI(model="gpt-4o-mini")


@tool
async def search(query: str, top_k: int, force_web: bool = False) -> any:
    """
    Search for information relevant to the given query.
    Use this tool to search basis, documents, and web to answer questions of the user.

    It will first search the cached documents. If there is no relevant information found, it will search the web and index the results before returning.

    Args:
        query: The query to search for.
        top_k: The number of documents to return. Use `5` or lower for easy, short answers. Use `10` or lower for more detailed answers. Use `20` or lower for the most detailed answers. It will be clipped to `20` if it exceeds.
        force_web: An optional boolean flag to force the search to be performed on the web. Defaults to `False`. Set it to `True` if you want to search the web always, ignoring the pre-indexed documents.

    Returns:
        The information that is relevant to the given query.

    Note:
        When providing a query, do not pass the original query. Instead, generate a refined query, following these guidelines:

        1. Identify and retain key concepts from the original query
        2. Remove unnecessary words or phrases
        3. Add relevant synonyms or related terms
        4. Consider the context of the search (academic, technical, general, etc.)
        5. Ensure the refined query is concise yet comprehensive

        You should prefer English for the refined query, but:
        - If the context is better in non-English, deduce the best language for the refined query and use it.
        - For example, you might prefer to use Japanese for the refined query if the context is better in Japanese, e.g. when searching for Japanese anime.
        - Or, you might prefer to use place names in native language, e.g. when searching for a specific location.
        - Whatever the language you choose, translate the original query to it if necessary.

        Here are some good examples (line-by-line, assume the current context is best in English):

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
        top_k = max(3, min(20, top_k))

        async def search_cached_documents(query: str) -> list[Document]:
            return [] if force_web else await document_storage.query(query, top_k)

        async def search_web(query: str) -> list[str]:
            results = await web_search.ainvoke(query)
            return [result["link"] for result in results]

        documents, urls = await asyncio.gather(
            search_cached_documents(query),
            search_web(query),
        )

        if len(documents) < top_k:
            now = datetime.now().isoformat()

            async def process_web_result(url: str):
                try:
                    content, metadata = await extract_html(url, llm)

                    if content == "":
                        return

                    document = Document(
                        page_content=content,
                        metadata=metadata,
                    )
                    chunks = document_splitter.split_documents([document])
                    chunks = [format_chunk(chunk, now) for chunk in chunks]
                    await document_storage.add_documents(chunks)

                except:
                    pass

            await asyncio.gather(*[process_web_result(url) for url in urls])

            extra_documents = await document_storage.query(
                query, top_k - len(documents)
            )
            documents.extend(extra_documents)

        if len(documents) == 0:
            return {
                "query": query,
                "documents": [],
                "warning": "no relevant information found",
            }

        return {
            "query": query,
            "documents": serialize_documents(documents),
        }

    except Exception as e:
        print(f"[search] warning: failed to search `{query}`: {e}")

        return {
            "status": "error",
            "cause": str(e),
        }


@tool
async def index_urls(urls: list[str]) -> any:
    """
    Index the given URLs.

    It is useful when user explicitly provides URLs to index, due to the fact that the web search is not always reliable.

    Args:
        urls: The URLs to index.

    Returns:
        It returns indexed urls or status and cause if it fails.
    """
    try:
        now = datetime.now().isoformat()

        async def process_url(url: str) -> str | None:
            try:
                content, metadata = await extract_html(url, llm)

                if content == "":
                    print(
                        f"[index_urls] warning: failed to extract `{url}`: content is empty"
                    )
                    return None

                document = Document(
                    page_content=content,
                    metadata=metadata,
                )
                chunks = document_splitter.split_documents([document])
                chunks = [format_chunk(chunk, now) for chunk in chunks]
                await document_storage.add_documents(chunks)

                return url

            except Exception as e:
                print(f"[index_urls] warning: failed to extract `{url}`: {e}")
                return None

        indexed_urls = await asyncio.gather(*[process_url(url) for url in urls])
        indexed_urls = [url for url in indexed_urls if url is not None]

        return indexed_urls

    except Exception as e:
        print(f"[index_urls] warning: failed to index `{urls}`: {e}")

        return {
            "status": "error",
            "cause": str(e),
        }


def serialize_documents(docs: list[Document]) -> list[str]:
    return [doc.page_content for doc in docs]


def format_chunk(chunk: Document, now: str) -> Document:
    minified_metadata = {
        **chunk.metadata,
    }

    del minified_metadata["url"]
    del minified_metadata["title"]
    del minified_metadata["description"]

    url = chunk.metadata["url"]
    title = chunk.metadata["title"]
    description = chunk.metadata["description"]

    if 1024 < len(url):
        url = url[:1024]

    if 1024 < len(title):
        title = title[:1024]

    if 1024 < len(description):
        description = description[:1024]

    return Document(
        page_content="\n".join(
            [
                "Timestamp:",
                now,
                "",
                "Url:",
                url,
                "",
                "Title:",
                title,
                "",
                "Description:",
                description,
                "",
                "Metadata:",
                json.dumps(minified_metadata, indent=2, ensure_ascii=False),
                "",
                "Content:",
                chunk.page_content,
            ]
        ),
        metadata={
            "timestamp": now,
            **chunk.metadata,
        },
    )
