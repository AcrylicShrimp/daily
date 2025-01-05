import os

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank


class DocumentStorage:
    def __init__(self, top_k_search: int = 20, top_k_rerank: int = 5):
        self.top_k_search = top_k_search
        self.top_k_rerank = top_k_rerank
        self.embeddings = VoyageAIEmbeddings(model="voyage-3-lite", batch_size=32)
        self.rerank = VoyageAIRerank(model="rerank-2-lite", top_k=self.top_k_rerank)
        self.vector_store = Chroma(
            collection_name="documents",
            embedding_function=self.embeddings,
            persist_directory=os.getenv("CHROMA_DIR"),
        )

    def add_documents(self, documents: list[Document]):
        self.vector_store.add_documents(documents)

    async def aadd_documents(self, documents: list[Document]):
        await self.vector_store.aadd_documents(documents)

    def query(self, query: str) -> list[Document]:
        searched = self.vector_store.search(
            query,
            "similarity_score_threshold",
            k=self.top_k_search,
            score_threshold=0.25,
        )
        reranked = self.rerank.compress_documents(searched, query)
        return reranked

    async def aquery(self, query: str) -> list[Document]:
        searched = await self.vector_store.asearch(
            query,
            "similarity_score_threshold",
            k=self.top_k_search,
            score_threshold=0.25,
        )
        reranked = await self.rerank.acompress_documents(searched, query)
        return reranked
