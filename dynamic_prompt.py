from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Pass API key explicitly
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1",
    task="text-generation",
    huggingfacehub_api_token=api_key
)

model = ChatHuggingFace(llm=llm)

st.header('Research Tool')

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

template = PromptTemplate(
    template="""Please summarize the research paper titled "{paper}" with the following specifications:
    Style: {style}
    Length: {length}

    1. Mathematical details:
    - Include relevant mathematical equations if present in the paper.
    - Explain the mathematical concepts using simple, intuitive code snippets where applicable.

    2. Analogies:
    - Use relatable analogies to simplify complex ideas.

    If certain information is not available in the paper, respond with:
    "Insufficient information available" instead of guessing.

    Ensure the summary is clear, accurate, and aligned with the provided style and length.
    """,
    input_variables=['paper', 'style', 'length']
)

prompt = template.invoke({
    'paper': paper,
    'style': style,
    'length': length
})

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)
