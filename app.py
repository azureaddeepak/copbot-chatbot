# app.py - CopBotChatbox: Chennai Police Assistant (GEMINI MODE - SECURE)

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

# Initialize session state for police station search
if "show_police_search" not in st.session_state:
    st.session_state.show_police_search = False

if "searched_area" not in st.session_state:
    st.session_state.searched_area = ""

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

# Custom CSS for design
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .sidebar .sidebar-content { background: #1f77b4; color: white; }
    .stButton>button { background-color: #1f77b4; color: white; }
    .stTextInput>div>div>input { border: 2px solid #1f77b4; }
</style>
""", unsafe_allow_html=True)

# Add logo and title + buttons in columns
col1, col2 = st.columns([1, 4])

with col1:
    st.image("tn_logo.png", width=90)  # Slightly larger logo

with col2:
    st.title("👮 Chennai District Police Assistance Bot")
    st.markdown("## Police Assistance Cell")
    st.markdown("### 👋 Welcome! I am the Chennai District Police Assistance bot. How can I help you?")

    # Buttons row
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

    with btn_col1:
        if st.button("🚨 Emergency Contacts", use_container_width=True):
            st.info("""
📞 **Police**: 100  
📞 **Women Helpline**: 1091  
📞 **Cyber Crime**: 1930  
📞 **Child Helpline**: 1098  
📞 **Ambulance/Fire**: 112
""")

    with btn_col2:
        if st.button("👮 Police Stations", use_container_width=True):
            st.info("📍 **Interactive Map Coming Soon**\n\nMeanwhile, use 'Find Nearby Police Station' to search by area.")

    with btn_col3:
        if st.button("📝 How to File Complaint?", use_container_width=True):
            st.info("""
👉 **Online**: Visit [Tamil Nadu Police Portal](https://www.tnpolice.gov.in) → Click 'e-FIR' or 'Complaint'\n
👉 **Offline**: Visit nearest police station → Submit written complaint → Get stamped copy\n
👉 **Documents Needed**: ID Proof, Address Proof, Incident Details, Photos/Videos (if any)
""")

with btn_col4:
    # Toggle button to show/hide search
    if st.button("📍 Find Nearby Police Station", use_container_width=True):
        st.session_state.show_police_search = not st.session_state.show_police_search

    # Show search box if toggled on
    if st.session_state.show_police_search:
        st.markdown("### 🔍 Search Police Station by Area")
        area = st.text_input(
            "Enter your area (e.g., Kovur, Velachery, Teynampet):",
            value=st.session_state.searched_area,
            key="area_input_unique",
            placeholder="Type your locality and press Enter..."
        )

        # Update session state when user types
        st.session_state.searched_area = area

        if area.strip():
            area_clean = area.strip().title()
            if area_clean in chennai_police_stations:
                station = chennai_police_stations[area_clean]
                st.success(f"""
**📍 {station['name']}**
**🏠 Address:** {station['address']}
**📞 Phone:** {station['phone']}
**🗺️ Jurisdiction:** {station['jurisdiction']}
""")
            else:
                st.warning(f"⚠️ No exact match for '{area_clean}'. Try these nearby areas:")
                suggestions = list(chennai_police_stations.keys())[:5]
                for loc in suggestions:
                    st.write(f"🔹 **{loc}** → {chennai_police_stations[loc]['name']}")

# Language toggle in sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Police_India.svg/1200px-Police_India.svg.png", width=100)
    st.title("👮 CopBotChatbox")
    st.markdown("### Chennai District Police")
    language = st.radio("Select Language / மொழியைத் தர்ந்தெடுக்கவும்", ["English", "தமிழ் (Tamil)"], index=0)
    st.markdown("---")

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
            google_api_key=st.secrets["GEMINI_API_KEY"],
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
st.caption("ℹ️ Official demo by Chennai District Police. Powered by Google Gemini. Responses based only on official documents.")
