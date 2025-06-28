# childmedia_rag_app.py
import streamlit as st
import torch
import re
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# Predefined questions and responses
song_questions = [
    "Does this song contain any violent themes, such as references to guns, killing, or physical aggression?",
    "Are there any explicit lyrics or bad words used in this song that might be considered offensive or inappropriate?",
    "Is the overall content of this song suitable for children, considering its themes, language, and messages?",
    "Does this song explicitly mention weapons, such as guns, knives, or other similar items?",
    "Are the messages conveyed in this song positive and uplifting for children?",
    "Does this song include any sexual content, references to sexual behavior, or suggestive language?",
    "Does this song offer any educational value, such as teaching the alphabet, basic math, or other learning content?",
    "Does this song promote emotional resilience and social skills among children?"
]

yes_responses = [
    "Yes, this song contains violent themes and is not suitable for children.",
    "Yes, this song includes explicit lyrics or bad words inappropriate for young audiences.",
    "No, the overall content is not suitable for children as it includes mature themes.",
    "Yes, this song explicitly mentions weapons which could be disturbing for children.",
    "Yes, the messages are positive and uplifting, beneficial for children.",
    "Yes, this song includes sexual content inappropriate for a child-friendly environment.",
    "Yes, this song offers significant educational value for children.",
    "Yes, this song promotes emotional resilience essential for children's development."
]

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.responses = []

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model with caching
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
    return tokenizer, model

# Processing functions
def process_song(song):
    song_new = re.sub(r'[\n]', ' ', song)
    return [song_new.replace("\'", "")]

def aggregate_embeddings(input_ids, attention_masks):
    mean_embeddings = []
    for input_id, mask in zip(input_ids, attention_masks):
        input_ids_tensor = torch.tensor([input_id]).to(DEVICE)
        mask_tensor = torch.tensor([mask]).to(DEVICE)
        
        with torch.no_grad():
            outputs = bert_model(input_ids_tensor, attention_mask=mask_tensor)
            word_embeddings = outputs.last_hidden_state.squeeze(0)
            
            valid_embeddings_mask = mask != 0
            valid_embeddings = word_embeddings[valid_embeddings_mask, :]
            mean_embedding = valid_embeddings.mean(dim=0)
            mean_embeddings.append(mean_embedding.unsqueeze(0))
    
    return torch.cat(mean_embeddings)

@st.cache_data(show_spinner=False)
def text_to_emb(list_of_text, max_input=512):
    data_token_index = tokenizer.batch_encode_plus(
        list_of_text, 
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=max_input,
        return_tensors="pt"
    )
    embeddings = aggregate_embeddings(
        data_token_index['input_ids'],
        data_token_index['attention_mask']
    )
    return embeddings

def RAG_QA(embeddings_questions, embeddings_song, n_responses=3):
    dot_product = embeddings_questions @ embeddings_song.T
    dot_product = dot_product.reshape(-1)
    sorted_indices = torch.argsort(dot_product, descending=True).tolist()
    return [yes_responses[i] for i in sorted_indices[:n_responses]]

# Load models
tokenizer, bert_model = load_model()

# Precompute question embeddings once
question_embeddings = text_to_emb(song_questions)

# Streamlit UI
st.set_page_config(
    page_title="Child Media Safety Checker",
    page_icon="🧒",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stTextArea textarea {
        min-height: 300px;
    }
    .metric-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .safe {
        color: #2ecc71;
        font-weight: bold;
    }
    .unsafe {
        color: #e74c3c;
        font-weight: bold;
    }
    .stProgress > div > div > div {
        background-color: #3498db;
    }
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.title("🧒 Child Media Content Safety Analyzer")
st.markdown("""
<div style='border-bottom: 1px solid #ddd; padding-bottom: 10px; margin-bottom: 30px;'>
    <p>Evaluate song lyrics for child-appropriate content using AI-powered analysis</p>
</div>
""", unsafe_allow_html=True)

# Main columns layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Input Lyrics")
    song_text = st.text_area(
        "Paste song lyrics below:", 
        height=350,
        placeholder="Enter or paste lyrics here...\n\nExample:\n\"Sunny day\nSweepin' the clouds away\nOn my way to where the air is sweet\""
    )
    
    analyze_btn = st.button(
        "🔍 Analyze Content", 
        type="primary",
        use_container_width=True,
        disabled=not song_text.strip()
    )

with col2:
    st.header("ℹ️ How It Works")
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
        <ol style='padding-left: 20px;'>
            <li>Paste song lyrics in the input box</li>
            <li>Click "Analyze Content"</li>
            <li>AI evaluates against 8 safety criteria</li>
            <li>Get instant safety assessment</li>
        </ol>
        <p>The system checks for:</p>
        <ul style='padding-left: 20px;'>
            <li>Violence and weapon references</li>
            <li>Explicit language</li>
            <li>Sexual content</li>
            <li>Age-appropriate themes</li>
            <li>Educational value</li>
        </ul>
        <p style='font-size: 0.85em; margin-top: 20px;'>
        Note: This tool uses BERT language model for content analysis. 
        Results should be verified by human reviewers.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Process lyrics when button is clicked
if analyze_btn and song_text.strip():
    with st.spinner("Analyzing content - this may take 10-20 seconds..."):
        progress_bar = st.progress(0)
        
        # Step 1: Preprocess lyrics
        processed_song = process_song(song_text)
        progress_bar.progress(25)
        
        # Step 2: Generate embeddings
        song_embeddings = text_to_emb(processed_song)
        progress_bar.progress(60)
        
        # Step 3: Run RAG analysis
        responses = RAG_QA(question_embeddings, song_embeddings)
        progress_bar.progress(85)
        
        # Store results in session state
        st.session_state.responses = responses
        st.session_state.processed = True
        progress_bar.progress(100)

# Display results if processed
if st.session_state.processed:
    st.header("📊 Safety Assessment Results")
    
    # Safety metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        violence = "Detected" if "violent" in st.session_state.responses[0] else "None"
        violence_class = "unsafe" if "violent" in st.session_state.responses[0] else "safe"
        st.markdown(f"""
        <div class='metric-box'>
            <h3>Violence</h3>
            <h2 class='{violence_class}'>{violence}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        explicit = "Detected" if "explicit" in st.session_state.responses[0] else "None"
        explicit_class = "unsafe" if "explicit" in st.session_state.responses[0] else "safe"
        st.markdown(f"""
        <div class='metric-box'>
            <h3>Explicit Language</h3>
            <h2 class='{explicit_class}'>{explicit}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        overall = "Unsuitable" if "not suitable" in st.session_state.responses[0] else "Suitable"
        overall_class = "unsafe" if "not suitable" in st.session_state.responses[0] else "safe"
        st.markdown(f"""
        <div class='metric-box'>
            <h3>Overall Safety</h3>
            <h2 class='{overall_class}'>{overall}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed findings
    st.subheader("🔍 Detailed Findings")
    for i, response in enumerate(st.session_state.responses, 1):
        st.info(f"**Finding #{i}:** {response}")
    
    # Disclaimer
    st.warning("**Disclaimer:** This AI assessment is not 100% accurate. Always combine with human judgment for content moderation decisions.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #777; font-size: 0.9rem; padding-top: 20px;'>
    <p>Child Media Safety Analyzer • Powered by BERT and Streamlit</p>
    <p>For educational purposes only</p>
</div>
""", unsafe_allow_html=True)