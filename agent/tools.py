import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate  
from langchain.embeddings import OpenAIEmbeddings
import wolframalpha

def ingest(path="data/*.txt"):
    loader = TextLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("GOOGLE_API_KEY"))
    vectordb = Weaviate.from_documents(chunks, embeddings, url=os.getenv("RAG_VECTOR_DB_URL"))
    return vectordb

def rag_tool(query, vectordb):
    docs = vectordb.similarity_search(query, k=5)
    return "\n\n".join(d.page_content for d in docs)

def wolfram_tool(expr):
    client = wolframalpha.Client(os.getenv("WOLFRAM_APP_ID"))
    res = client.query(expr)
    return next(res.results).text
