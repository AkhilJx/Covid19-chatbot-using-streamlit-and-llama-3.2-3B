"""
This code implements a COVID-19 FAQ chatbot using the LangChain library.

1. **Imports and Setup**:
   - Various modules from LangChain are imported for document loading, embeddings, vector storage, and large language model (LLM) functionality.
   - All warnings are suppressed to keep the output clean.

2. **Model and Vector Database Initialization**:
   - An embedding model (`HuggingFaceInstructEmbeddings`) is initialized with the `hkunlp/instructor-large` model.
   - The vector database path is set to `"faiss_index"` for storing the FAISS vector store.
   - The `Ollama` model (`llama3.2:latest`) is instantiated with a callback manager for real-time output streaming.

3. **Vector Database Creation**:
   - The `create_vector_db()` function checks for the existence of the vector database. If not found, it:
     - Loads data from a CSV file (`covid_faq.csv`), specifically from the "questions" column.
     - Creates a FAISS vector store instance using the loaded data and saves it locally.

4. **Chat Functionality**:
   - The `chat_with_me()` function sets up chat interaction by:
     - Loading the existing vector database.
     - Creating a retriever to fetch the most relevant documents based on user queries.

5. **Prompt Template Definition**:
   - A `PromptTemplate` is defined to guide the LLM in generating context-based answers while avoiding fabrication if no answer is found.

6. **Chain Creation**:
   - A `RetrievalQA` chain is established, connecting the LLM and retriever for generating responses based on user queries and the retrieved context.

Overall, this code provides the functionality for a chatbot that can effectively respond to COVID-19 related questions using a pre-loaded FAQ dataset.
"""

import os
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

#define the embedding model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
#define the vectordb file path
vectordb_file_path = "faiss_index"
#define the llm
# llm = OllamaLLM(model="llama3.2:latest")
llm = Ollama(model="llama3.2:latest", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

def create_vector_db():
    # Check if the vector database file already exists
    if not os.path.exists(vectordb_file_path) or len(os.listdir(vectordb_file_path)) == 0:
        print("Creating vector database...")
        # Load data from FAQ sheet
        loader = CSVLoader(file_path='covid_faq.csv', source_column="questions")
        data = loader.load()

        # Create a FAISS instance for vector database from 'data'
        vectordb = FAISS.from_documents(documents=data,
                                        embedding=instructor_embeddings)

        # Save vector database locally
        vectordb.save_local(vectordb_file_path)
    else:
        print("Vector database already exists. Skipping creation.")

def chat_with_me():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7, search_kwargs={'k': 1})

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
       In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
       If the answer is not found in the context, kindly state "I don't know. This is Covid related Chatbot" Don't try to make up an answer.

       CONTEXT: {context}

       QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain
