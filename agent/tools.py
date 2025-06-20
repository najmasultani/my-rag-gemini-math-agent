import os
import glob
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.embeddings import OpenAIEmbeddings
import wolframalpha

def ingest(path="data/*.pdf"):
    all_docs = []
    for pdf_path in glob.glob(path):
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

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
