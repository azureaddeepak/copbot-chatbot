# app.py - CopBotChatbox: chennai Police Assistant (GEMINI MODE - SECURE)

import streamlit as st
import pandas as pd
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Set page config
st.set_page_config(
    page_title="👮 CopBotChatbox - chennai Police",
    page_icon="👮",
    layout="wide"
)

# Custom CSS for design
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .sidebar .sidebar-content { background: #1f77b4; color: white; }
    .stButton>button { background-color: #1f77b4; color: white; }
    .stTextInput>div>div>input { border: 2px solid #1f77b4; }
</style>
""", unsafe_allow_html=True)
# Add logo and title
col1, col2 = st.columns([1, 4])
with col1:
    st.image("tn_logo.png", width=60)  # Small logo

# Language toggle in sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Police_India.svg/1200px-Police_India.svg.png", width=100)
    st.title("👮 CopBotChatbox")
    st.markdown("### chennai District Police")
    language = st.radio("Select Language / மொழியைத் தேர்ந்தெடுக்கவும்", ["English", "தமிழ் (Tamil)"], index=0)
    st.markdown("---")
    st.markdown("### 📍 Police Stations")
    st.write("Map coming soon...")
    st.markdown("### 🆘 Emergency Numbers")
    st.write("📞 Police: 100")
    st.write("📞 Women Helpline: 1091")
    st.write("📞 Cyber Crime: 1930")

# Main Header
if language == "English":
    st.title("👮 Welcome to Chennai District Police Assistance Bot")
    st.markdown("Ask me anything about filing complaints, FIRs, procedures, or emergency contacts.")
else:
    st.title("👮 தூத்துக்குடி மாவட்ட காவல்துறை உதவி போட் க்கு வரவேற்கிறோம்")
    st.markdown("புகார் பதிவு, எஃப்ஐஆர், நடைமுறைகள் அல்லது அவசர தொடர்புகள் குறித்து என்னிடம் கேளுங்கள்.")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# Load data from Excel (only once)
@st.cache_data
def load_data():
    df = pd.read_excel("Chatbot_Data.xlsx")
    return df

if not st.session_state.data_loaded:
    with st.spinner("Loading official police data... (Powered by Gemini)"):
        df = load_data()
        # Prepare text for RAG
        df['combined_text'] = (
            "Category: " + df['Category'].astype(str) + " | " +
            "Subcategory: " + df['Subcategory'].astype(str) + " | " +
            "Question: " + df['User_Query'].astype(str) + " | " +
            "Answer: " + df['Chatbot_Response'].astype(str) + " | " +
            "Keywords: " + df['Keywords'].astype(str) + " | " +
            "Source: " + df['Source_Document_Section'].astype(str)
        )
        text_data = df['combined_text'].tolist()
        full_text = "\n".join(text_data)

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(full_text)

        # Create local embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embeddings)

        st.session_state.vectorstore = vectorstore
        st.session_state.data_loaded = True
        st.success("✅ Official Police Knowledge Base Loaded!")

# Chat interface
st.markdown("### 💬 Ask Your Question")

user_query = st.text_input(
    "Type your question here..." if language == "English" else "உங்கள் கேள்வியை இங்கே தட்டச்சு செய்யவும்...",
    placeholder="E.g., How to file FIR?" if language == "English" else "எ.கா., எஃப்ஐஆர் எப்படி பதிவு செய்வது?"
)

if user_query and st.session_state.vectorstore:
    with st.spinner("🤔 CopBot is thinking... (Gemini AI)"):
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

        template = """You are 'CopBot', the official AI assistant of Thoothukudi District Police.
        Answer the question based ONLY on the context provided.
        If unsure, say "I cannot answer based on official data."
        Keep response clear, concise, and citizen-friendly.
        Respond in the SAME LANGUAGE as the question.

        Context: {context}

        Question: {question}

        Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        
        # ✅ SECURE: Use API key from Streamlit Secrets
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=st.secrets["GEMINI_API_KEY"],  # ← Securely loaded
            temperature=0,
            convert_system_message_to_human=True
        )
        
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(user_query)

        st.markdown("### 🤖 CopBot Response:")
        st.info(response)

# Footer
st.markdown("---")
st.caption("ℹ️ Official demo by Thoothukudi District Police. Powered by Google Gemini. Responses based only on official documents.")
