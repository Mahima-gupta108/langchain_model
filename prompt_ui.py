from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import os

# Load .env locally (ignored on Streamlit Cloud)
load_dotenv()

# Get HuggingFace API key from environment or Streamlit secrets
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

# Initialize HuggingFace LLM with API key
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1",
    task="text-generation",
    huggingfacehub_api_token=api_key  # pass the API key here
)

model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.header("Research Tool")
user_input = st.text_input("Enter your prompt here")

if st.button('Summarize') and user_input:
    result = model.invoke(user_input)
    st.write(result.content)


