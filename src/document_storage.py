import os

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank


class DocumentStorage:
    def __init__(self):
        self.embeddings = VoyageAIEmbeddings(model="voyage-3-lite", batch_size=32)
        self.rerank = VoyageAIRerank(model="rerank-2-lite")
        self.vector_store = Chroma(
            collection_name="documents",
            embedding_function=self.embeddings,
            persist_directory=os.getenv("CHROMA_DIR"),
        )

    async def add_documents(self, documents: list[Document]):
        await self.vector_store.aadd_documents(documents)

    async def query(self, query: str, top_k: int) -> list[Document]:
        searched = await self.vector_store.asearch(
            query,
            "mmr",
            k=min(10, top_k * 3),
            fetch_k=min(40, top_k * 6),
            lambda_mult=0.6,
        )
        reranked = await self.rerank.acompress_documents(searched, query)

        if top_k < len(reranked):
            reranked = reranked[:top_k]

        return reranked
