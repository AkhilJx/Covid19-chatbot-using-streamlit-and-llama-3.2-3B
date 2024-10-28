# Covid19-Chatbot-using-streamlit-and-llama-3.2-3B
A Chatbot that provides COVID-related answers to the user based on the user prompt.


The chatbot uses Llama 3.2-3B LLM Model to process the query given to the chatbot by the user.  

The data about the COVID-related questions is given in the form of a CSV file. The CSV file is covid_faq.csv and uploaded in the repo.

The CSV file contains the questions and answers as columns.

The Embedding model used for embedding the contents in the CSV is a hugging face model named "hkunlp/instructor-large".

The vector DB used in this project is FAISS.

# Steps Invloved:

