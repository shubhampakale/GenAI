from dotenv import load_dotenv #type: ignore
load_dotenv() # Load environment variables from .env file

import asyncio
from langchain_community.document_loaders import PyPDFLoader #type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter #type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings #type: ignore
import os
import pinecone  # type: ignore
from typing import Any, Tuple
from typing import Optional
# Use Pinecone client directly instead of LangChain's Pinecone wrapper


async def main():
    # 1. Initialize Loader and Splitter
    loader = PyPDFLoader("Dsa.pdf")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

    # 2. Async Load: Get list[Document]
    docs = await loader.aload()
    print(f"Total documents loaded: {len(docs)}")
    # 3. Async Split: Get chunked list[Document]
    # atransform_documents is the async equivalent for splitting docs
    chunked_docs = await text_splitter.atransform_documents(docs)
    print
    #print(f"Total chunks created: {len(chunked_docs)}")
    
    # 5. Initialize Pinecone (modern SDK)
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

    if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
        raise RuntimeError("Set PINECONE_API_KEY and PINECONE_INDEX_NAME in .env to use Pinecone")

    # New Pinecone SDK: create a Pinecone client instance
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    print(f"Pinecone index '{PINECONE_INDEX_NAME}' initialized via Pinecone client.")

    # Embed chunks using embeddings and upsert to Pinecone directly
    texts = [getattr(c, "page_content", "") for c in chunked_docs]
    vectors = None
    # prefer async API, fall back to sync if needed
    if hasattr(embeddings, "aembed_documents"):
        vectors = await embeddings.aembed_documents(texts)
    elif hasattr(embeddings, "embed_documents"):
        vectors = await asyncio.to_thread(embeddings.embed_documents, texts)
    else:
        raise RuntimeError("Embeddings client has no embed_documents method")

    to_upsert = []
    for i, vec in enumerate(vectors):
        meta = {}
        if hasattr(chunked_docs[i], "metadata"):
            try:
                meta = chunked_docs[i].metadata or {}
            except Exception:
                meta = {}
        # include chunk text in metadata if missing
        chunk_text = getattr(chunked_docs[i], "page_content", None)
        if chunk_text and not (isinstance(meta, dict) and meta.get("text")):
            if not isinstance(meta, dict):
                meta = {}
            meta["text"] = chunk_text
        to_upsert.append((str(i), vec, meta))

    # Upsert into Pinecone index
    pinecone_index.upsert(vectors=to_upsert)
    print(f"Upserted {len(to_upsert)} vectors to Pinecone index {PINECONE_INDEX_NAME}")




if __name__ == "__main__":
    asyncio.run(main())
