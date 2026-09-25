from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_voyageai import VoyageAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from huggingface_hub import login

import os
import key_param
    
login(key_param.HF_TOKEN)

dbName = "book_mongodb_chunks"
collectionName = "chunked_data"
index = "vector_index"

vectorStore = MongoDBAtlasVectorSearch.from_connection_string(
    key_param.MONGODB_URI,
    dbName + "." + collectionName,
    VoyageAIEmbeddings(voyage_api_key=key_param.VOYAGE_API_KEY, model="voyage-3.5-lite"),
    index_name=index,
)

os.environ["GOOGLE_API_KEY"] = key_param.GEMINI_API_KEY
gemini = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0

)

def query_data(query: str):
    retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,#amount of documents to return
            "pre_filter": { "hasCode": { "$eq": False } },
            "score_threshold": 0.01
        }
    )

    template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Do not answer the question if there is no given context.
        Do not answer the question if it is not related to the context.
        Do not give recommendations to anything other than the house architectural design.
        Context:
        {context}
        Question: {question}
        """

    custom_rag_prompt = PromptTemplate.from_template(template)
    retrieve = {
            "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), 
            "question": RunnablePassthrough()
            }

    llm = gemini

    response_parser = StrOutputParser()

    rag_chain = (
        retrieve
        | custom_rag_prompt
        | llm
        | response_parser
    )

    answer = rag_chain.invoke(query)
    return answer

query = "Is there any restriction about yellow undertones in the design?"
answer = query_data(query)
print(answer) 

 