"""RAG Retrieval and Query Execution Module."""

import logging
import sys

from huggingface_hub import login
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_voyageai import VoyageAIEmbeddings

import key_param

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rag_retrieval")

# Environment setup
# os.environ["GOOGLE_API_KEY"] = key_param.GEMINI_API_KEY
# login(key_param.HF_TOKEN)


# 1. Initialize Vector Store Connection
DB_NAME = "rag_example"
COLLECTION_NAME = "chunked_data"
INDEX_NAME = "vector_index"

embeddings = VoyageAIEmbeddings(
    voyage_api_key=key_param.VOYAGE_API_KEY, model="voyage-3.5-lite"
)

vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=key_param.MONGODB_URI,
    namespace=f"{DB_NAME}.{COLLECTION_NAME}",
    embedding=embeddings,
    index_name=INDEX_NAME,
)


# 2. Configure Retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
        "pre_filter": {"hasCode": {"$eq": False}},
    },
)

# 3. Define Prompt Template
SYSTEM_PROMPT = """You are a helpful assistant answering questions based solely on the provided context.

Rules:
1. Answer using ONLY the information in the Context section below.
2. If the answer is not contained in the context, explicitly state "I do not have enough information to answer."
3. Do not assume or extrapolate beyond the provided text.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

# 4. Initialize LLM & Chain
gemini = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=0.0,
)
llm = gemini  # you can switch to other LLMs


def format_docs(docs):
    """Concatenate retrieved document chunks into a formatted string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# 5. Execution query RAG function
def query_rag(question: str) -> str:
    """Execute vector search and return synthesized answer from LLM."""
    logger.info("Querying RAG system with question: '%s'", question)

    retrieve = {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    rag_chain = retrieve | prompt | llm | StrOutputParser()

    return rag_chain.invoke(question)


# change here for a question based on the PDF that fed the RAG system.
query = "Is query_rag any restriction about yellow undertones in the design?"
answer = query_rag(query)
print(answer)
