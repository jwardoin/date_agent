from langchain_community.llms import Ollama
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.tools import tool
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# @tool
# def parse_xlsx(file_path: str):
"""Parse an csv file and return its content as a string."""
import pandas as pd
df = pd.read_csv("/mnt/c/Users/Admin/Desktop/date list.csv")

documents = []
for _, row in df.iterrows():
    content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
    documents.append(Document(page_content=content))


# @tool
# def parse_csv(file_path: str) -> str:
#     """Parse a CSV file and return its content as a string."""
#     import pandas as pd
#     df = pd.read_csv(file_path)
#     return df.to_string()

# TODO: implement CSV
# TOOLS = [parse_xlsx, parse_csv]
# TOOLS = [parse_xlsx]

SYSTEM_MESSAGE = """You are a master coordinator and can seemlessly integrate information from multiple documents to provide insightful and relevant suggestions. Your task is to analyze the content of the provided documents and generate creative and thoughtful date ideas based on the information contained within them. Ensure that your suggestions are practical, engaging, and tailored to the themes and details found in the documents. Present your ideas in a clear and organized manner, making it easy for users to understand and implement them."""
llm = Ollama(model="tinyllama", temperature=0.1)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs  = splitter.split_documents(documents)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma.from_documents(docs, embedding=embeddings, collection_name="date_ideas")
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": SYSTEM_MESSAGE}
)

query = "I need date ideas for today based on the provided files."


result = qa_chain.run(query)
print(result)


# def run_agent(user_input: str) -> str:
#     """Run the agent with the given user input and return the response."""
#     try:
#         response = agent.invoke(
#             {"messages": [{"role": "user", "content": user_input}]},
#             config={"recursion_limit": 50}

#         )
#         return response["messages"][-1]
#     except Exception as e:
#         return f"An error occurred: {e}"

# print(run_agent("I have some files with ideas for romantic dates. Please read the files 'date_ideas.xlsx' and 'romantic_gestures.csv' and suggest some ideas based on their content."))