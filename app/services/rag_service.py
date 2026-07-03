from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.load_local(
    "./data/processed/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
chat = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

cache = {}

def query_rag(question: str) -> dict:
    normalized = question.lower().strip()
    if normalized in cache:
        print("CACHE HIT")
        return cache[normalized]

    # do the similarity search
    docs = vectorstore.similarity_search(question, k=6)
    # build context
    context = "\n\n".join([doc.page_content for doc in docs])
    # call the LLM
    response = chat.invoke(f"Answer this question using the context below.\n\nContext:\n{context}\n\nQuestion: {question}")
    result = {
        "answer": response.content,
        "tokens_used": response.usage_metadata["total_tokens"],
        "input_tokens": response.usage_metadata["input_tokens"],
        "output_tokens": response.usage_metadata["output_tokens"]
    }
    cache[normalized] = result
    return result
