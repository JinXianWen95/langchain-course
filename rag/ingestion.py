import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()
os.environ["LANGSMITH_PROJECT"] = "rag_gist_ingestion"


if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader(
        r"C:\Users\jinxw\projects\langchain-agentic\langchain-course\rag\mediumblog1.txt",
        encoding="UTF-8",
    )
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = PineconeEmbeddings(model="llama-text-embed-v2")

    print("ingesting")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
