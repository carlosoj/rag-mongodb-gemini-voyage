from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_voyageai import VoyageAIEmbeddings
import key_param

dbName = "book_mongodb_chunks"
collectionName = "chunked_data"
index = "vector_index"

vectorStore = MongoDBAtlasVectorSearch.from_connection_string(
    key_param.MONGODB_URI,
    dbName + "." + collectionName,
    VoyageAIEmbeddings(voyage_api_key=key_param.VOYAGE_API_KEY, model="voyage-3.5-lite"),
    index_name=index,
)

def query_data(query: str):
    retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,
            "pre_filter": { "hasCode": { "$eq": False } },
            "score_threshold": 0.01
        }
    )
    results = retriever.invoke(query)
    return results

query = "Is there any restriction about yellow undertones in the design?"
results = query_data(query)

print(results) 