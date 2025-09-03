from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not api_key:
    st.error("API key not found! Set HUGGINGFACEHUB_API_TOKEN in .env or environment variables.")
    st.stop()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1",
    task="text-generation",
    huggingfacehub_api_token=api_key  
)

model = ChatHuggingFace(llm=llm)

st.header("Research Tool")
user_input = st.text_input("Enter your prompt here")

if st.button('Summarize') and user_input:
   
    result = model.invoke(user_input)
    st.write(result.content)








