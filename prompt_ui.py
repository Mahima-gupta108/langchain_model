from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

import os

load_dotenv() 

api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")  

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1",
    task="text-generation",
    
    )
model=ChatHuggingFace(llm=llm)
st.header("research tool")
user_input=st.text_input("enter your prompt here")
if st.button('summarize'):
    result=model.invoke(user_input)
    st.write(result.content)
