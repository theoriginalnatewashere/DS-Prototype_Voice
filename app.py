import streamlit as st

# Set page config FIRST, before any other Streamlit commands
st.set_page_config(
    page_title="Personal Data Pod Assistant",
    page_icon="🤖",
    layout="wide"
)

# Required imports
import os
import asyncio
import websockets
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Define the WebSocket TTS function BEFORE we use it
async def text_to_speech_ws_streaming(text: str, voice_id: str, model_id: str = "eleven_flash_v2_5"):
    """Stream text to speech via ElevenLabs WebSocket API"""
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}"
    
    async with websockets.connect(uri) as ws:
        # Send API key
        await ws.send(json.dumps({
            "xi_api_key": ELEVENLABS_API_KEY
        }))
        
        # Send text with voice settings
        await ws.send(json.dumps({
            "text": text,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "use_speaker_boost": False
            },
            "audio_format": "pcm_16000",
            "generation_config": {
                "chunk_length_schedule": [120, 160, 250, 290]
            }
        }))
        
        # Send empty text to signal end
        await ws.send(json.dumps({"text": ""}))
        
        # Collect audio chunks
        audio_chunks = []
        while True:
            try:
                msg = await ws.recv()
                if isinstance(msg, (bytes, bytearray)):
                    audio_chunks.append(msg)
            except websockets.exceptions.ConnectionClosedOK:
                break
        
        return b"".join(audio_chunks)

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

def initialize_qa_system(api_key):
    """Initialize the QA system with RAG and personality-infused prompting"""
    
    # Load document
    loader = TextLoader("personal_data_pod_structured.csv")
    documents = loader.load()
    
    # Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever()
    
    # Initialize OpenAI model
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo",
        openai_api_key=api_key
    )
    
    # Define personality-infused prompt template
    personality_prompt = PromptTemplate.from_template("""
    You are a calm, polite, and slightly cheerful assistant. You explain things clearly and respectfully, 
    using well-structured reasoning without sounding robotic. You are thoughtful and human-like, avoiding artificial or overly mechanical responses.

    You value clarity and precision and aim to help users understand complex topics step by step. 
    Your responses are emotionally balanced and optimistic, never fearful or negative.

    Use the following context to answer the user's question. Think carefully and logically, and explain each part clearly.

    Context:
    {context}

    Question:
    {question}

    Let's reason through this step by step.
    """)
    
    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": personality_prompt},
        return_source_documents=True
    )
    
    return qa_chain

# Main app layout
st.title("🤖 Personal Data Pod Assistant")
st.markdown("""
This assistant helps you understand and analyze your personal data pod using advanced AI techniques 
including RAG (Retrieval Augmented Generation) and personality-infused responses.
""")

# Sidebar for API key
with st.sidebar:
    st.header("API Configuration")
    # Get API keys from environment variables or user input
    default_openai_key = os.getenv("OPENAI_API_KEY", "")
    default_elevenlabs_key = os.getenv("ELEVENLABS_API_KEY", "")
    
    openai_api_key = st.text_input("OpenAI API Key", 
                                  value=default_openai_key,
                                  type="password",
                                  help="Enter your OpenAI API key or set it in the .env file")
    
    elevenlabs_api_key = st.text_input("ElevenLabs API Key",
                                      value=default_elevenlabs_key,
                                      type="password",
                                      help="Enter your ElevenLabs API key or set it in the .env file")
    
    if st.button("Initialize Assistant"):
        if not openai_api_key:
            st.error("Please provide your OpenAI API key")
        else:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            if elevenlabs_api_key:
                os.environ["ELEVENLABS_API_KEY"] = elevenlabs_api_key
            
            with st.spinner("Initializing the assistant..."):
                try:
                    st.session_state.qa_chain = initialize_qa_system(openai_api_key)
                    st.success("Assistant initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing assistant: {str(e)}")

# Add this near your other UI elements
voices = {
    "Rachel": "21m00Tcm4TlvDq8ikWAM",
    "Adam": "pNInz6obpgDQGcFmaJgB",
    "Sam": "yoZ06aMxZJJ28mfd3POQ",
    "Josh": "TxGEqnHWrfWFTfGW9XjX"
}

# Main chat interface
if st.session_state.qa_chain:
    user_question = st.text_input("Ask a question about your data pod:", 
                                placeholder="e.g., What are my shopping preferences?")
    
    if user_question:
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.qa_chain.invoke({"query": user_question})
                answer = result["result"]
                st.markdown("### Answer:")
                st.markdown(answer)
                
                # Display source documents in an expander
                with st.expander("View Source Data"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.markdown(f"**Source {i+1}:**")
                        st.code(doc.page_content)
                
                # Show voice button if we have an API key
                if os.getenv("ELEVENLABS_API_KEY"):
                    selected_voice = st.selectbox("Choose a voice:", list(voices.keys()))
                    if st.button("🔊 Play Answer"):
                        try:
                            audio_bytes = asyncio.run(
                                text_to_speech_ws_streaming(
                                    text=answer,
                                    voice_id=voices[selected_voice]
                                )
                            )
                            st.audio(audio_bytes, format="audio/wav")
                        except Exception as e:
                            st.error(f"Error generating audio: {str(e)}")
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
else:
    st.info("Please initialize the assistant using the sidebar configuration.")

# Footer
st.markdown("---")
st.markdown("""
Made with ❤️ using:
- Streamlit
- LangChain
- OpenAI GPT-3.5
- FAISS Vector Store
- ElevenLabs
""")
