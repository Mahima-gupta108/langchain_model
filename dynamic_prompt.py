from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
load_dotenv() 

api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN") 
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1",
    task="text-generation",
     
)
model=ChatHuggingFace(llm=llm)
st.header('research tool')
paper = st.selectbox(
    "Choose a research paper:", 
    [
        "Attention Is All You Need (2017, Transformer)",
        "BERT: Bidirectional Encoder Representations (2018)",
        "GPT-2: Generative Pre-trained Transformer (2019)",
        "GPT-3: Language Models are Few-Shot Learners (2020)",
        "DALL·E: Zero-Shot Text-to-Image Generation (2021)",
        "Stable Diffusion: High-Resolution Image Synthesis (2022)",
        "PaLM: Pathways Language Model (2022)",
        "LLaMA: Large Language Model Meta AI (2023)"
    ]
)
style = st.selectbox("Choose the explanation style:", 
                     ["Beginner Friendly", "Technical", "Intermediate"])

length = st.selectbox("Choose the length of response:", 
                      ["Short (100-200 words)", "Medium (200-400 words)", "Long (400+ words)"])
template=PromptTemplate(
    template="""please summarize the research paper titled "{paper}" with the following specifications:
    style:{style}
    length:{length}
    1.mathematical details:
    -include relevant mathematical equation if present in the paper.
    -explain the mathematical concepts using simple,intiutive code snippets where applicable.
    2.analogies:
    -use reletable analogies to simplify complex ideas.
    if certain information is not available in the paper,respond with:"insufficient information available" instead of guessing.
    ensure the summary is clear,accurate and aligned with the provided style and length
    """,
    input_variables=['paper','style','length']
)
prompt=template.invoke({
    'paper':{paper},
    'style':{style},
    'length':{length}
})
if st.button('summarize'):
    result=model.invoke(prompt)
    st.write(result.content)