!pip install -U langchain langchain-community sentence-transformers beautifulsoup4 chromadb gradio

import os
import requests
from bs4 import BeautifulSoup
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import Chroma


url = "https://brainlox.com/courses/category/technical" 
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

content = " ".join([p.text for p in soup.find_all("p")]) # extracts text from the webpage


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) # Creats embeddings by using Hugging Face 
texts = text_splitter.split_text(content)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # Loading Hugging Face embeddings 
vector_db = Chroma.from_texts(texts, embeddings)

# Creating a Gradio Interface for chatbot
def chatbot(query):
    docs = vector_db.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])

# Launching Gradio UI
gr.Interface(fn=chatbot, inputs="text", outputs="text", title="LangChain Chatbot (Free)").launch(share=True)
