from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
loader = PyPDFLoader(
    file_path="/Users/varunutukuri/Desktop/rag-doc-intel/data/raw/Varun_Utukuri_Resume.pdf",

)

pages = loader.load()
print(type(pages))
print(len(pages))
print(pages[0])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = text_splitter.split_documents(pages)
print(f"Number of chunks: {len(chunks)}")
print(chunks[0])

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("./data/processed/faiss_index")
print("FAISS index saved successfully")