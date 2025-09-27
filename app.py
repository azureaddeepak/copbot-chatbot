import streamlit as st
import pandas as pd
import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
import requests
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="👮 CopBotChatbox - Chennai Police",
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

# Language toggle in sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Police_India.svg/1200px-Police_India.svg.png", width=100)
    st.title("👮 CopBotChatbox")
    st.markdown("### Chennai District Police")
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
    st.title("👮 சென்னை மாவட்ட காவல்துறை உதவி போட் க்கு வரவேற்கிறோம்")
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
        st.session_state.text_splitter = text_splitter
        chunks = text_splitter.split_text(full_text)

        # Use SentenceTransformer directly (CPU-safe)
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            st.session_state.model = model
            st.write("🧠 Encoding knowledge base...")
            vectors = model.encode(chunks, show_progress_bar=False, device="cpu")

            # Create FAISS index
            dimension = vectors.shape[1]
            from faiss import IndexFlatL2
            index = IndexFlatL2(dimension)
            index.add(vectors.astype('float32'))

            # Wrap into LangChain-compatible retriever
            class SimpleVectorStore:
                def __init__(self, index, texts, model):
                    self.index = index
                    self.texts = texts
                    self.model = model

                def similarity_search(self, query, k=3):
                    query_vec = self.model.encode([query], device="cpu").astype('float32')
                    distances, indices = self.index.search(query_vec, k)
                    results = [self.texts[i] for i in indices[0]]
                    return [{"page_content": r} for r in results]

            st.session_state.vectorstore = SimpleVectorStore(index, chunks, model)

            # Define custom web search tool
            @tool
            def duckduckgo_search(query: str) -> str:
                """Search the web using DuckDuckGo and return top results."""
                if not BS4_AVAILABLE:
                    return "Web search not available: BeautifulSoup not installed."
                try:
                    url = f"https://duckduckgo.com/html/?q={query}"
                    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                    soup = BeautifulSoup(response.text, 'html.parser')
                    results = soup.find_all('a', class_='result__a')
                    snippets = [result.get_text() for result in results[:5]]
                    return '\n'.join(snippets)
                except Exception as e:
                    return f"Search failed: {e}"

            # Create agent with web search tool
            if BS4_AVAILABLE:
                try:
                    llm_agent = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        google_api_key=st.secrets["GEMINI_API_KEY"],
                        temperature=0,
                        convert_system_message_to_human=True
                    )
                    st.session_state.agent = initialize_agent(
                        tools=[duckduckgo_search],
                        llm=llm_agent,
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        verbose=False
                    )
                except Exception as e:
                    st.session_state.agent = None
            else:
                st.session_state.agent = None

            st.session_state.data_loaded = True
            st.success("✅ Official Police Knowledge Base Loaded with Agentic Capabilities!")

        except Exception as e:
            st.error(f"❌ Failed to load embeddings: {e}")
            st.stop()

# Chat interface
st.markdown("### 💬 Ask Your Question")

user_query = st.text_input(
    "Type your question here..." if language == "English" else "உங்கள் கேள்வியை இங்கே தட்டச்சு செய்யவும்...",
    placeholder="E.g., How to file FIR?" if language == "English" else "எ.கா., எஃப்ஐஆர் எப்படி பதிவு செய்வது?"
)

if user_query and st.session_state.vectorstore:
    with st.spinner("🤔 CopBot is thinking... (AI)"):
        search_result = ""
        if st.session_state.agent:
            # Use agent to search web
            search_result = st.session_state.agent.run(f"Search the web for information related to: {user_query}")

            # Add search result to vectorstore
            if search_result.strip():
                chunks_new = st.session_state.text_splitter.split_text(search_result)
                vectors_new = st.session_state.model.encode(chunks_new, show_progress_bar=False, device="cpu")
                st.session_state.vectorstore.index.add(vectors_new.astype('float32'))
                st.session_state.vectorstore.texts.extend(chunks_new)

        # Retrieve from vectorstore (updated if agent used)
        docs = st.session_state.vectorstore.similarity_search(user_query, k=3)
        context = "\n".join([doc["page_content"] for doc in docs])

        template = """You are 'CopBot', the official AI assistant of Chennai District Police.
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
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        try:
            response = chain.invoke(user_query)
            st.markdown("### 🤖 CopBot Response:")
            st.info(response)
        except Exception as e:
            st.error(f"❌ Error generating response: {e}")

# Footer
st.markdown("---")
st.caption("ℹ️ Official demo by Chennai District Police. Powered by Google Gemini. Responses based only on official documents.")
