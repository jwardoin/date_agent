from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

import pandas as pd
df = pd.read_csv(os.getenv("CSV_FILE_PATH"))

documents = []
for _, row in df.iterrows():
    content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
    documents.append(Document(page_content=content))

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs  = splitter.split_documents(documents)


SYSTEM_MESSAGE = """You are a master coordinator and can seemlessly integrate information from multiple documents to provide insightful and relevant suggestions. Your task is to analyze the content of the provided documents and generate creative and thoughtful date ideas based on the information contained within them. Ensure that your suggestions are practical, engaging, and tailored to the themes and details found in the documents. Present your ideas in a clear and organized manner, making it easy for users to understand and implement them."""


embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma.from_documents(docs, embedding=embeddings, collection_name="date_ideas")
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_MESSAGE), 
                                           ("human", "Context:\n{context}\n\nQuestion: {question}")])

llm = OllamaLLM(model=os.getenv("OLLAMA_MODEL"), temperature=0.1)
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)

query = "I need date ideas for today based on the provided files."

result = rag_chain.invoke(query)
print(result)
