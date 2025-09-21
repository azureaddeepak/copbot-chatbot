import streamlit as st
import pandas as pd
import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# Chennai Police Station Data (City-Wide Coverage)
chennai_police_stations = {
    # Central Chennai
    "Parry's Corner": {
        "name": "Parry's Corner Police Station",
        "address": "No. 1, Rajaji Salai, George Town, Chennai - 600001",
        "phone": "044-2522 1000",
        "jurisdiction": "George Town, Parry's, Sowcarpet"
    },
    "Royapettah": {
        "name": "Royapettah Police Station",
        "address": "No. 3, Pycrofts Road, Royapettah, Chennai - 600014",
        "phone": "044-2847 1000",
        "jurisdiction": "Royapettah, Triplicane, Chepauk"
    },
    "Egmore": {
        "name": "Egmore Police Station",
        "address": "No. 1, Pantheon Road, Egmore, Chennai - 600008",
        "phone": "044-2819 1000",
        "jurisdiction": "Egmore, Kilpauk, Chetpet"
    },
    "Teynampet": {
        "name": "Teynampet Police Station",
        "address": "No. 1, GN Chetty Road, Teynampet, Chennai - 600018",
        "phone": "044-2833 1000",
        "jurisdiction": "Teynampet, Nandanam, Raja Annamalai Puram"
    },
    "Mylapore": {
        "name": "Mylapore Police Station",
        "address": "No. 1, Royapettah High Road, Mylapore, Chennai - 600004",
        "phone": "044-2499 1000",
        "jurisdiction": "Mylapore, Santhome, Adyar"
    },
    "Thiruvanmiyur": {
        "name": "Thiruvanmiyur Police Station",
        "address": "No. 1, Rajiv Gandhi Salai, Thiruvanmiyur, Chennai - 600041",
        "phone": "044-2444 1000",
        "jurisdiction": "Thiruvanmiyur, Adyar, Besant Nagar"
    },

    # South Chennai
    "Velachery": {
        "name": "Velachery Police Station",
        "address": "Velachery Bypass Road, Chennai - 600042",
        "phone": "044-2255 1000",
        "jurisdiction": "Velachery, Madipakkam, Pallikaranai"
    },
    "Pallikaranai": {
        "name": "Pallikaranai Police Station",
        "address": "No. 1, Pallikaranai Main Road, Chennai - 600100",
        "phone": "044-2266 1000",
        "jurisdiction": "Pallikaranai, Keelkattalai, Kovilambakkam"
    },
    "Tambaram": {
        "name": "Tambaram Police Station",
        "address": "Tambaram Sanatorium, Chennai - 600045",
        "phone": "044-2239 1000",
        "jurisdiction": "Tambaram, Chromepet, Pallavaram"
    },
    "Pallavaram": {
        "name": "Pallavaram Police Station",
        "address": "No. 1, Grand Southern Trunk Road, Pallavaram, Chennai - 600043",
        "phone": "044-2268 1000",
        "jurisdiction": "Pallavaram, Chromepet, Tirusulam"
    },
    "Kovur": {
        "name": "Kovur Police Station",
        "address": "No. 1, GST Road, Kovur, Chennai - 600127",
        "phone": "044-2766 1000",
        "jurisdiction": "Kovur, Perungalathur, Vandalur"
    },
    "Vandalur": {
        "name": "Vandalur Police Station",
        "address": "No. 1, Vandalur Main Road, Chennai - 600048",
        "phone": "044-2747 1000",
        "jurisdiction": "Vandalur, Mudichur, Uthandi"
    },

    # West Chennai
    "Anna Nagar": {
        "name": "Anna Nagar Police Station",
        "address": "No. 3, 3rd Avenue, Anna Nagar, Chennai - 600040",
        "phone": "044-2616 1000",
        "jurisdiction": "Anna Nagar, Villivakkam, Padi"
    },
    "Villivakkam": {
        "name": "Villivakkam Police Station",
        "address": "No. 1, EVR Periyar Salai, Villivakkam, Chennai - 600049",
        "phone": "044-2668 1000",
        "jurisdiction": "Villivakkam, Kolathur, Peravallur"
    },
    "Ambattur": {
        "name": "Ambattur Police Station",
        "address": "No. 1, Jawaharlal Nehru Road, Ambattur, Chennai - 600053",
        "phone": "044-2626 1000",
        "jurisdiction": "Ambattur, Avadi, Pattabiram"
    },
    "Avadi": {
        "name": "Avadi Police Station",
        "address": "No. 1, Avadi Road, Avadi, Chennai - 600054",
        "phone": "044-2655 1000",
        "jurisdiction": "Avadi, Pattabiram, Thiruninravur"
    },

    # North Chennai
    "Tondiarpet": {
        "name": "Tondiarpet Police Station",
        "address": "No. 1, EVR Periyar Salai, Tondiarpet, Chennai - 600081",
        "phone": "044-2591 1000",
        "jurisdiction": "Tondiarpet, Royapuram, Washermanpet"
    },
    "Royapuram": {
        "name": "Royapuram Police Station",
        "address": "No. 1, Old Jail Road, Royapuram, Chennai - 600013",
        "phone": "044-2533 1000",
        "jurisdiction": "Royapuram, Tondiarpet, Basin Bridge"
    },
    "Perambur": {
        "name": "Perambur Police Station",
        "address": "No. 1, Perambur High Road, Chennai - 600011",
        "phone": "044-2671 1000",
        "jurisdiction": "Perambur, Kolathur, Peravallur"
    },
    "Thiru Vi Ka Nagar": {
        "name": "Thiru Vi Ka Nagar Police Station",
        "address": "No. 1, P H Road, Thiru Vi Ka Nagar, Chennai - 600019",
        "phone": "044-2642 1000",
        "jurisdiction": "Thiru Vi Ka Nagar, Perambur, Kolathur"
    }
}

# Set page config
st.set_page_config(
    page_title="👮 CopBotChatbox - Chennai Police",
    page_icon="👮",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .sidebar .sidebar-content { background: #1f77b4; color: white; }
    .stButton>button { background-color: #1f77b4; color: white; }
    .stTextInput>div>div>input { border: 2px solid #1f77b4; }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([1, 4])
with col1:
    try:
        st.image("tn_logo.png", width=90)
    except:
        st.write("🖼️ Logo")
with col2:
    st.title("👮 Chennai District Police Assistance Bot")

# Sidebar
with st.sidebar:
    try:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Police_India.svg/1200px-Police_India.svg.png", width=100)
    except:
        st.write("🖼️ Police Icon")
    st.title("👮 CopBotChatbox")
    st.markdown("### Chennai District Police")
    language = st.radio("Select Language / மொழியைத் தர்ந்தெடுக்கவும்", ["English", "தமிழ் (Tamil)"], index=0)
    st.markdown("---")
    try:
        st.image("gandhi.jpg", use_container_width=True, caption="Mahatma Gandhi")
    except:
        st.caption("Mahatma Gandhi")

tab1, tab2, tab3, tab4 = st.tabs(["Home", "🚨 Emergency Contacts", "📝 How to File Complaint?", "👮 Nearby Police Stations"])

with tab1:
    # Welcome
    st.markdown("## Police Assistance Cell")
    st.markdown("### 👋 Welcome! I am the Chennai District Police Assistance bot. How can I help you?")

# Load data from Excel (only once)
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("Chatbot_Data.xlsx")
        return df
    except Exception as e:
        st.error(f"❌ Could not load Chatbot_Data.xlsx: {e}")
        return None

if not st.session_state.data_loaded:
    with st.spinner("Loading official police data... (Powered by Gemini)"):
        df = load_data()
        if df is None:
            st.stop()

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

        # Use SentenceTransformer directly (CPU-safe)
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
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
            st.session_state.data_loaded = True
            st.success("✅ Official Police Knowledge Base Loaded!")

        except Exception as e:
            st.error(f"❌ Failed to load embeddings: {e}")
            st.stop()

    # Chat interface
    st.markdown("### 💬 Ask Your Question")

    user_query = st.text_input(
        "Type your question here..." if language == "English" else "உங்கள் கேள்வியை இங்கே தட்டச்சு செய்யவும்...",
        placeholder="E.g., How to file FIR?" if language == "English" else "எ.கா., எஃப்ஐஆர் எப்படி பதிவு செய்வது?"
    )

    if user_query and hasattr(st.session_state, 'vectorstore') and st.session_state.vectorstore:
        if user_query.lower() in ["hi", "hello", "hey"]:
            st.markdown("### 🤖 CopBot Response:")
            st.success("Hello! I am the Chennai District Police Assistance Bot. How can I help you today?")
        else:
            with st.spinner("🤔 CopBot is thinking... (Gemini AI)"):
                # Simulate retriever
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
                try:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        google_api_key=st.secrets["GEMINI_API_KEY"],
                        temperature=0,
                        convert_system_message_to_human=True
                    )
                except Exception as e:
                    st.error(f"❌ Gemini API error: {e}")
                    st.stop()

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

with tab2:
    st.markdown("## 🚨 Emergency Contacts")
    st.info("""
📞 **Police**: 100
📞 **Women Helpline**: 1091
📞 **Cyber Crime**: 1930
📞 **Child Helpline**: 1098
📞 **Ambulance/Fire**: 112
""")

with tab3:
    st.markdown("## 📝 How to File Complaint?")
    st.info("""
👉 **Online**: Visit [Tamil Nadu Police Portal](https://www.tnpolice.gov.in) → Click 'e-FIR' or 'Complaint'\n
👉 **Offline**: Visit nearest police station → Submit written complaint → Get stamped copy\n
👉 **Documents Needed**: ID Proof, Address Proof, Incident Details, Photos/Videos (if any)
""")

with tab4:
    st.markdown("## 👮 Nearby Police Stations - Chennai")
    st.markdown("### 🔍 Search Police Station by Area")

    # Search Box
    area_input = st.text_input("Enter area in Chennai (e.g., Velachery, Kovur)", placeholder="e.g., Velachery, Kovur, Chennai")

    if st.button("🔍 Search", key="search_btn"):
        if area_input.strip():
            area_clean = area_input.strip().title()
            if area_clean in chennai_police_stations:
                station = chennai_police_stations[area_clean]
                st.success(f"Found: {station['name']}")
                st.markdown(
                    f"""
                    <div style="background-color:#d4edda; padding:15px; border-radius:8px; margin-top:10px; border-left:5px solid #28a745;">
                        <strong>📍 {station['name']}</strong><br>
                        <strong>🏠 Address:</strong> {station['address']}<br>
                        <strong>📞 Phone:</strong> {station['phone']}<br>
                        <strong>🗺️ Jurisdiction:</strong> {station['jurisdiction']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                matches = [loc for loc in chennai_police_stations.keys() if area_input.lower().strip() in loc.lower()]
                if matches:
                    station = chennai_police_stations[matches[0]]
                    st.warning(f"No exact match. Showing closest: {station['name']}")
                    st.markdown(
                        f"""
                        <div style="background-color:#fff3cd; padding:15px; border-radius:8px; margin-top:10px; border-left:5px solid #ffc107;">
                            <strong>📍 {station['name']}</strong><br>
                            <strong>🏠 Address:</strong> {station['address']}<br>
                            <strong>📞 Phone:</strong> {station['phone']}<br>
                            <strong>🗺️ Jurisdiction:</strong> {station['jurisdiction']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.error("No police station found for that area. Try nearby areas like Velachery, Teynampet, etc.")
        else:
            st.warning("Please enter an area to search.")


# Footer
st.markdown("---")
st.caption("ℹ️ Official demo by Chennai District Police. Powered by Google Gemini. Responses based only on official documents.")
