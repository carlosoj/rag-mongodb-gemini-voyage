## Basic RAG
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain_voyageai import VoyageAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List
from langchain_core.documents import Document
from langchain_community.document_transformers.openai_functions import (
    create_metadata_tagger,
)

import os
import key_param

# Set the MongoDB URI, DB, Collection Names
    
os.environ["GOOGLE_API_KEY"] = key_param.GEMINI_API_KEY
client = MongoClient(key_param.MONGODB_URI)
dbName = "book_mongodb_chunks"
collectionName = "chunked_data"
collection = client[dbName][collectionName]

loader = PyPDFLoader(key_param.SOURCE_FILE_PATH)
pages = loader.load()
cleaned_pages = []

for page in pages:
    if len(page.page_content.split(" ")) > 20:
        cleaned_pages.append(page)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)

class PageMetadata(BaseModel):
    title: str = Field(description="Title of the document section")
    keywords: List[str] = Field(description="Key topics discussed")
    hasCode: bool = Field(description="Whether the text contains code snippets")

schema = {
    "properties": {
        "title": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "hasCode": {"type": "boolean"},
    },
    "required": ["title", "keywords", "hasCode"],
}

chat_gpt = ChatOpenAI(
    api_key= key_param.LLM_API_KEY, temperature = 0, model="gpt-3.5-turbo"
)
gemini = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0

)
structured_llm = gemini.with_structured_output(PageMetadata)

document_transformer = create_metadata_tagger(metadata_schema=schema, llm=gemini)

#docs = document_transformer.transform_documents(cleaned_pages)
docs = []
for page in cleaned_pages:
    try:
        metadata_res = structured_llm.invoke(page.page_content)
        # Update page metadata with extracted dictionary
        page.metadata.update(metadata_res.model_dump())
    except Exception as e:
        print(f"Failed to extract metadata for page: {e}")
    docs.append(page)

split_docs = text_splitter.split_documents(docs)

embeddings = VoyageAIEmbeddings(voyage_api_key=key_param.VOYAGE_API_KEY, model="voyage-3.5-lite")

vectorStore = MongoDBAtlasVectorSearch.from_documents(
    split_docs, embeddings, collection=collection
)