# Covid19-Chatbot-using-streamlit-and-llama-3.2-3B
A Chatbot that provides COVID-related answers to the user based on the user prompt.


The chatbot uses Llama 3.2-3B LLM Model to process the query given to the chatbot by the user.  

The data about the COVID-related questions is given in the form of a CSV file. The CSV file is covid_faq.csv and uploaded in the repo.

The CSV file contains the questions and answers as columns.

The Embedding model used for embedding the contents in the CSV is a hugging face model named "hkunlp/instructor-large".

The vector DB used in this project is FAISS.

# Steps Involved:

1. Import the required libraries
2. Define the embedding model, vector db filename and the llm to be used.
3. load the data from the CSV file using CSVLoader function.
4. Create a FAISS instance for a vector database from the above data using FAISS.from_documents by passing the data as document and embedding model as embedding in the argument.
5. Save vector database locally
6. Steps 3 to 5 are done only once if the vector db doesn't exist.
7. To chat with the chatbot based on the data provided, first Load the vector database from the local folder
8. Create a retriever for querying the vector database
9. Define a prompt template.
10. Create a prompt to be given to the LLM based on the prompt template by using the function PromptTemplate. The prompt template and input variables are given as parameters to the function PromptTemplate.
11. Retrieve the content from the data using RetrievalQA.from_chain_type function by passing the llm, chain_type, retriever, input_key (user query), return_source_documents value, and chain_type_kwargs as "prompt": PROMPT (step 10 value)
12. Provide an app to the user using Streamlit.
13. When the user enters a query and clicks the submit button, steps 7 to 11 get executed and finally display the output.
14. if the user ask anything outside the data, the chatbot will say "I don't know. This is Covid related Chatbot" as it is mentioned in the prompt.

# Output 1:
![Screenshot from 2024-10-28 12-16-08](https://github.com/user-attachments/assets/3cd7f81c-a94e-4de2-b8f3-5a0f111dfc4d)

# Output 2:
![Screenshot from 2024-10-28 12-19-05](https://github.com/user-attachments/assets/d1058d9b-5f21-4b9c-82fb-29583d5c6ded)

# ......................................................................................................
![image](https://github.com/user-attachments/assets/a6d364a9-e871-403d-918f-615779b5baec)

![image](https://github.com/user-attachments/assets/c52ee6b6-9419-4066-b5d2-387367126c03)




