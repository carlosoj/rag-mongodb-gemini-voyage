## RAG ingest
import logging
import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from pydantic import BaseModel, Field, ConfigDict
from pymongo import MongoClient

import key_param

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rag_ingest")
os.environ["GOOGLE_API_KEY"] = key_param.GEMINI_API_KEY


class PageMetadata(BaseModel):
    """Schema for document chunk metadata extraction."""

    model_config = ConfigDict(extra="allow")  # Ignores/allows unexpected output keys

    title: str = Field(description="Title of the document section")
    keywords: list[str] = Field(description="Key topics discussed in the chunk")
    hasCode: bool = Field(description="Whether the text contains code snippets")


def extract_meaningful_content_from_pdf(
    file_path: str, min_word_count: int = 20
) -> list[Document]:
    """Load PDF file and filter out low-content pages (e.g., blank, cover, TOC)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found at: {file_path}")

    logger.info("Loading PDF document from: %s", file_path)
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    cleaned_pages = []

    for page in pages:
        if len(page.page_content.split(" ")) > min_word_count:
            cleaned_pages.append(page)
    logger.info(
        "Loaded %d pages (Filtered out %d short/empty pages)",
        len(cleaned_pages),
        len(pages) - len(cleaned_pages),
    )
    return cleaned_pages


# 1 Database Setup
client = MongoClient(key_param.MONGODB_URI)
dbName = "rag_example"
collectionName = "chunked_data"
collection = client[dbName][collectionName]

# Clear existing entries to prevent duplicates during re-runs
collection.delete_many({}) 

# 2 Get useful content from the PDF and filter out low-content pages
raw_pages = extract_meaningful_content_from_pdf(key_param.SOURCE_FILE_PATH)

# 3 Create vector embeddings from the PDF and attach metadata to each chunk
gpt = ChatOpenAI(api_key=key_param.GPT_API_KEY, temperature=0, model="gpt-3.5-turbo")
gemini = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=0.0,
)

selected_llm = gemini  # You can switch here between GPT and Gemini

structured_llm = selected_llm.with_structured_output(PageMetadata)

docs = []
for page in raw_pages:
    try:
        metadata_res = structured_llm.invoke(page.page_content)
        # Add metadata to each chunk
        page.metadata.update(metadata_res.model_dump())
    except Exception as e:
        logger.error(f"Failed to extract metadata for page: {e}")
    docs.append(page)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)

embeddings = VoyageAIEmbeddings(
    voyage_api_key=key_param.VOYAGE_API_KEY, model="voyage-3.5-lite"
)

# 4 insert data
vectorStore = MongoDBAtlasVectorSearch.from_documents(
    split_docs, embeddings, collection=collection
)
