# app.py - Medical Note Simplifier with Doctor/Patient Modes
import streamlit as st
import torch
import json
import os
from datetime import datetime
from transformers import BartTokenizer, BartForConditionalGeneration
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from textstat import flesch_reading_ease
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Medical Note Simplifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SESSION STATE ====================

if 'user_mode' not in st.session_state:
    st.session_state.user_mode = None

# ==================== DICTIONARY MANAGEMENT ====================

DICTIONARY_FILE = 'medical_dictionary.json'

def load_dictionary():
    """Load medical dictionary from JSON file"""
    if os.path.exists(DICTIONARY_FILE):
        with open(DICTIONARY_FILE, 'r') as f:
            return json.load(f)
    else:
        # Default starter dictionary
        default_dict = {
            "hypertension": "high blood pressure",
            "hypotension": "low blood pressure",
            "tachycardia": "fast heart rate",
            "bradycardia": "slow heart rate",
            "dyspnea": "shortness of breath",
            "orthopnea": "difficulty breathing when lying down",
            "myocardial infarction": "heart attack",
            "MI": "heart attack",
            "CVA": "stroke",
            "cerebrovascular accident": "stroke",
            "edema": "swelling",
            "acute": "sudden",
            "chronic": "long-lasting",
            "bilateral": "both sides",
            "unilateral": "one side",
            "proximal": "closer to the body",
            "distal": "farther from the body",
            "prognosis": "expected outcome",
            "diagnosis": "identified condition",
            "thrombocytopenia": "low platelet count",
            "anemia": "low red blood cells",
            "leukocytosis": "high white blood cell count",
            "hyperglycemia": "high blood sugar",
            "hypoglycemia": "low blood sugar",
            "pneumonia": "lung infection",
            "COPD": "chronic lung disease",
            "CHF": "heart failure",
            "AF": "irregular heartbeat",
            "arrhythmia": "irregular heartbeat",
            "UTI": "urinary tract infection",
            "CKD": "chronic kidney disease",
            "hepatomegaly": "enlarged liver",
            "nephropathy": "kidney disease",
            "neuropathy": "nerve damage"
        }
        save_dictionary(default_dict)
        return default_dict

def save_dictionary(dictionary):
    """Save dictionary to JSON file"""
    with open(DICTIONARY_FILE, 'w') as f:
        json.dump(dictionary, f, indent=2, sort_keys=True)

def add_term_to_dictionary(medical_term, simple_explanation):
    """Add a new term to the dictionary"""
    dictionary = load_dictionary()
    dictionary[medical_term.lower().strip()] = simple_explanation.lower().strip()
    save_dictionary(dictionary)
    return True

# ==================== MODEL LOADING ====================

@st.cache_resource
def load_model():
    """Load BioBART model (cached to avoid reloading)"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "GanjinZero/biobart-base"
        
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        model = model.to(device)
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, "cpu"

# ==================== SUMMARIZATION FUNCTIONS ====================

def simplify_medical_terms(text):
    """Replace medical terms with simple explanations using word boundaries"""
    import re
    
    dictionary = load_dictionary()
    simplified = text
    
    # Sort by length (longest first) to avoid partial replacements
    sorted_terms = sorted(dictionary.items(), key=lambda x: len(x[0]), reverse=True)
    
    for medical, plain in sorted_terms:
        # Use word boundaries (\b) to match whole words only
        # This prevents "MI" from matching inside "mild", "admin", etc.
        pattern = r'\b' + re.escape(medical) + r'\b'
        simplified = re.sub(pattern, plain, simplified, flags=re.IGNORECASE)
    
    return simplified

def create_patient_summary(clinical_text, tokenizer, model, device):
    """
    Generate patient-friendly summary using hybrid approach:
    1. LexRank extraction
    2. BioBART rewriting  
    3. Medical term simplification
    """
    try:
        # Step 1: Extract key sentences with LexRank
        parser = PlaintextParser.from_string(clinical_text, Tokenizer("english"))
        lex_summarizer = LexRankSummarizer()
        key_sentences = lex_summarizer(parser.document, 10)
        extracted = ' '.join(str(s) for s in key_sentences)
        
        # Step 2: Generate summary with BioBART
        inputs = tokenizer(extracted, max_length=1024, truncation=True, 
                          return_tensors="pt").to(device)
        
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=250,
                min_length=80,
                num_beams=4,
                length_penalty=0.9,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Step 3: Simplify medical terms
        summary = simplify_medical_terms(summary)
        
        # Calculate readability
        readability = flesch_reading_ease(summary)
        
        return summary, readability
    
    except Exception as e:
        return f"Error processing text: {str(e)}", 0

# ==================== MODE SELECTION SCREEN ====================

if st.session_state.user_mode is None:
    
    # Center the content
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Title
    st.markdown("""
    <h1 style='text-align: center; color: #1f77b4;'>
        🏥 Medical Note Simplifier
    </h1>
    <h3 style='text-align: center; color: #666;'>
        Understanding Medical Information Made Easy
    </h3>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Mode selection
    st.markdown("<h3 style='text-align: center;'>Please select your role:</h3>", 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Patient button
        st.markdown("""
        <div style='padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; text-align: center; margin: 20px 0;'>
            <h2 style='color: white; margin: 0;'>👤 Patient</h2>
            <p style='color: white; margin: 10px 0;'>Get easy-to-understand explanations of your medical notes</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Continue as Patient", key="patient_btn", 
                     use_container_width=True, type="primary"):
            st.session_state.user_mode = "patient"
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Doctor button
        st.markdown("""
        <div style='padding: 30px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    border-radius: 15px; text-align: center; margin: 20px 0;'>
            <h2 style='color: white; margin: 0;'>⚕️ Healthcare Provider</h2>
            <p style='color: white; margin: 10px 0;'>Simplify notes for patients & improve the medical dictionary</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Continue as Healthcare Provider", key="doctor_btn", 
                     use_container_width=True, type="secondary"):
            st.session_state.user_mode = "doctor"
            st.rerun()
    
    # Disclaimer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.warning("""
    ⚠️ **Important Disclaimer:**  
    This tool is for **educational purposes only**. It does NOT provide medical diagnosis, 
    treatment recommendations, or replace consultation with healthcare professionals.  
    **Always consult your doctor for medical advice.**
    """)
    
    # Footer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #999; font-size: 14px;'>
        <p>IE 7500 Project | Northeastern University | Powered by BioBART & LexRank</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.stop()

# ==================== HEADER (AFTER MODE SELECTED) ====================

# Top bar with mode indicator and switch button
col1, col2 = st.columns([4, 1])

with col1:
    if st.session_state.user_mode == "patient":
        st.title("🏥 Medical Note Simplifier - 👤 Patient Mode")
    else:
        st.title("🏥 Medical Note Simplifier - ⚕️ Provider Mode")

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Switch Mode", use_container_width=True):
        st.session_state.user_mode = None
        st.rerun()

st.markdown("---")

# ==================== DOCTOR/PROVIDER MODE ====================

if st.session_state.user_mode == "doctor":
    
    tabs = st.tabs(["📝 Simplify Notes", "📚 Manage Dictionary", "📊 Statistics"])
    
    # ===== TAB 1: SIMPLIFY NOTES =====
    with tabs[0]:
        st.markdown("### Transform Clinical Notes into Patient-Friendly Summaries")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Input: Clinical Notes")
            
            uploaded_file = st.file_uploader(
                "Upload Medical Notes (TXT file)", 
                type=['txt'],
                key="doctor_upload"
            )
            
            clinical_text = st.text_area(
                "Or paste clinical notes here:",
                height=350,
                placeholder="Paste clinical notes, test results, or medical reports here...",
                key="doctor_text"
            )
            
            if uploaded_file:
                clinical_text = uploaded_file.read().decode('utf-8')
                st.success("✅ File uploaded successfully!")
            
            process_button = st.button(
                "🔄 Generate Patient Summary", 
                type="primary", 
                key="doctor_process",
                use_container_width=True
            )
        
        with col2:
            st.subheader("👤 Output: Patient-Friendly Summary")
            
            if process_button and clinical_text:
                with st.spinner("🔄 Processing clinical notes..."):
                    tokenizer, model, device = load_model()
                    
                    if model is not None:
                        summary, readability = create_patient_summary(
                            clinical_text, tokenizer, model, device
                        )
                        
                        # Display summary
                        st.markdown("#### Generated Summary:")
                        st.info(summary)
                        
                        # Readability analysis
                        st.markdown("#### 📊 Readability Analysis")
                        
                        col_r1, col_r2 = st.columns(2)
                        
                        with col_r1:
                            if readability >= 60:
                                st.success(f"✅ Flesch Score: **{readability:.1f}**")
                                level = "8-9th grade (Easy)"
                                color = "green"
                            elif readability >= 30:
                                st.warning(f"⚠️ Flesch Score: **{readability:.1f}**")
                                level = "College level (Moderate)"
                                color = "orange"
                            else:
                                st.error(f"❌ Flesch Score: **{readability:.1f}**")
                                level = "Graduate level (Difficult)"
                                color = "red"
                        
                        with col_r2:
                            st.metric("Reading Level", level)
                        
                        # Download button
                        st.markdown("---")
                        st.download_button(
                            label="📥 Download Patient Summary",
                            data=summary,
                            file_name=f"patient_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("Failed to load model. Please try again.")
            
            elif process_button:
                st.warning("⚠️ Please provide clinical notes first!")
    
    # ===== TAB 2: MANAGE DICTIONARY =====
    with tabs[1]:
        st.markdown("### 📚 Medical Term Dictionary Management")
        st.info("💡 Add new medical terms to help patients understand their notes better!")
        
        # Add new term section
        st.markdown("#### ➕ Add New Term")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            new_medical_term = st.text_input(
                "Medical Term:",
                placeholder="e.g., thrombocytopenia",
                key="new_term",
                help="Enter the complex medical term"
            )
        
        with col2:
            new_simple_term = st.text_input(
                "Patient-Friendly Explanation:",
                placeholder="e.g., low platelet count",
                key="new_simple",
                help="Enter a simple explanation anyone can understand"
            )
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            add_button = st.button("➕ Add Term", type="primary", use_container_width=True)
        
        # Handle adding new term
        if add_button:
            if new_medical_term and new_simple_term:
                dictionary = load_dictionary()
                
                if new_medical_term.lower() in dictionary:
                    st.warning(f"⚠️ Term '{new_medical_term}' already exists:")
                    st.code(f"{new_medical_term} → {dictionary[new_medical_term.lower()]}")
                    
                    if st.button("🔄 Update Definition"):
                        add_term_to_dictionary(new_medical_term, new_simple_term)
                        st.success(f"✅ Updated: '{new_medical_term}' → '{new_simple_term}'")
                        st.rerun()
                else:
                    add_term_to_dictionary(new_medical_term, new_simple_term)
                    st.success(f"✅ Added new term: '{new_medical_term}' → '{new_simple_term}'")
                    st.balloons()
                    
                    # Clear inputs
                    st.rerun()
            else:
                st.error("❌ Please fill in both fields!")
        
        st.markdown("---")
        
        # View and search dictionary
        st.markdown("#### 📖 Current Dictionary")
        
        dictionary = load_dictionary()
        
        # Search
        search_term = st.text_input(
            "🔍 Search dictionary:", 
            placeholder="Type to search medical terms or explanations...",
            key="search_dict"
        )
        
        if search_term:
            filtered_dict = {
                k: v for k, v in dictionary.items() 
                if search_term.lower() in k.lower() or search_term.lower() in v.lower()
            }
        else:
            filtered_dict = dictionary
        
        # Display dictionary
        if filtered_dict:
            st.caption(f"Showing {len(filtered_dict)} of {len(dictionary)} terms")
            
            # Convert to DataFrame for nice display
            import pandas as pd
            df_dict = pd.DataFrame([
                {"Medical Term": k.title(), "Simple Explanation": v.capitalize()} 
                for k, v in sorted(filtered_dict.items())
            ])
            
            st.dataframe(
                df_dict, 
                use_container_width=True, 
                height=400,
                hide_index=True
            )
            
            # Download dictionary
            st.markdown("---")
            dict_json = json.dumps(dictionary, indent=2)
            st.download_button(
                label="📥 Download Complete Dictionary (JSON)",
                data=dict_json,
                file_name=f"medical_dictionary_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        else:
            st.info("No terms found matching your search.")
    
    # ===== TAB 3: STATISTICS =====
    with tabs[2]:
        st.markdown("### 📊 Dictionary Statistics")
        
        dictionary = load_dictionary()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📚 Total Terms", len(dictionary))
        
        with col2:
            avg_term_length = sum(len(k) for k in dictionary.keys()) / len(dictionary)
            st.metric("📏 Avg Term Length", f"{avg_term_length:.1f} chars")
        
        with col3:
            avg_explanation = sum(len(v.split()) for v in dictionary.values()) / len(dictionary)
            st.metric("💬 Avg Explanation", f"{avg_explanation:.1f} words")
        
        with col4:
            longest = max(dictionary.items(), key=lambda x: len(x[0]))
            st.metric("🔤 Longest Term", f"{len(longest[0])} chars")
        
        st.markdown("---")
        
        # Recent additions (if we track timestamps)
        st.markdown("#### 📈 Dictionary Growth")
        st.success(f"Current dictionary contains **{len(dictionary)} medical terms**")
        st.info("💡 Tip: The more terms added by healthcare providers, the better the patient summaries become!")

# ==================== PATIENT MODE ====================

else:  # Patient mode
    
    # Patient-specific disclaimer
    st.warning("""
    ⚠️ **Important Information for Patients:**  
    
    This tool helps you understand medical terminology in your test results. However:
    - ❌ It does NOT diagnose medical conditions
    - ❌ It does NOT provide treatment advice
    - ❌ It does NOT replace your doctor's consultation
    
    **✅ DO:** Use this to better understand medical terms before talking to your doctor  
    **❌ DON'T:** Make medical decisions based solely on this tool  
    
    **Always discuss your results with your healthcare provider.**
    """)
    
    st.markdown("---")
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Your Medical Notes")
        st.caption("Upload or paste your medical test results or clinical notes")
        
        uploaded_file = st.file_uploader(
            "Upload Medical Notes (TXT file)", 
            type=['txt'],
            key="patient_upload",
            help="Upload a text file with your medical notes"
        )
        
        clinical_text = st.text_area(
            "Or paste your medical notes here:",
            height=350,
            placeholder="""Example:
            
Patient presents with elevated blood pressure reading of 165/95.
Laboratory results show glucose level of 145 mg/dL (fasting).
ECG demonstrates sinus tachycardia at 110 bpm.
Chest X-ray reveals bilateral infiltrates consistent with pneumonia.
            """,
            key="patient_text"
        )
        
        if uploaded_file:
            clinical_text = uploaded_file.read().decode('utf-8')
            st.success("✅ File uploaded successfully!")
        
        process_button = st.button(
            "🔄 Simplify My Medical Notes", 
            type="primary", 
            key="patient_process",
            use_container_width=True
        )
    
    with col2:
        st.subheader("👤 Easy-to-Understand Explanation")
        st.caption("Your medical notes translated into plain English")
        
        if process_button and clinical_text:
            with st.spinner("🔄 Creating your easy-to-read summary..."):
                tokenizer, model, device = load_model()
                
                if model is not None:
                    summary, readability = create_patient_summary(
                        clinical_text, tokenizer, model, device
                    )
                    
                    # Display summary in a nice box
                    st.markdown("#### Your Summary:")
                    st.success(summary)
                    
                    # Readability feedback
                    st.markdown("#### 📊 How Easy is This to Read?")
                    
                    if readability >= 60:
                        st.success(f"""
                        ✅ **Readability Score: {readability:.1f}**  
                        This summary is **easy to read** (8-9th grade level).  
                        Most people should understand this without difficulty.
                        """)
                    elif readability >= 30:
                        st.info(f"""
                        ℹ️ **Readability Score: {readability:.1f}**  
                        This summary is **moderately easy** (college level).  
                        You might need to read it carefully or ask your doctor about some parts.
                        """)
                    else:
                        st.warning(f"""
                        ⚠️ **Readability Score: {readability:.1f}**  
                        This summary is still **somewhat complex** (graduate level).  
                        Please discuss any unclear parts with your healthcare provider.
                        """)
                    
                    # Download option
                    st.markdown("---")
                    st.download_button(
                        label="📥 Download Summary",
                        data=summary,
                        file_name=f"my_medical_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.error("❌ Could not load the model. Please try again later.")
        
        elif process_button:
            st.warning("⚠️ Please upload a file or paste your medical notes first!")
    
    # Help section for patients
    st.markdown("---")
    
    with st.expander("❓ Need More Help Understanding Medical Terms?"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📚 Trusted Medical Resources:**
            - [MedlinePlus Medical Dictionary](https://medlineplus.gov/mplusdictionary.html)
            - [Mayo Clinic Medical Terms](https://www.mayoclinic.org/medical-terms)
            - [NIH Health Information](https://health.nih.gov/)
            - [CDC Health Topics](https://www.cdc.gov/health-topics.html)
            """)
        
        with col2:
            st.markdown("""
            **👨‍⚕️ Tips for Talking to Your Doctor:**
            - ✅ Write down questions before your appointment
            - ✅ Bring this summary with you
            - ✅ Ask your doctor to explain anything unclear
            - ✅ Request printed educational materials
            - ✅ Don't hesitate to ask for clarification
            """)
    
    # Dictionary stats for patients
    with st.expander("📖 About the Medical Dictionary"):
        dictionary = load_dictionary()
        st.info(f"""
        Our medical dictionary currently contains **{len(dictionary)} terms** 
        that have been simplified by healthcare providers to help patients 
        understand their medical notes.
        
        The dictionary is continuously growing as doctors add new terms!
        """)

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p style='margin: 5px;'><strong>Medical Note Simplifier</strong></p>
    <p style='margin: 5px; font-size: 14px;'>IE 7500 Neural NLP Project | Northeastern University</p>
    <p style='margin: 5px; font-size: 12px;'><em>Powered by Hybrid Summarization (LexRank + BioBART)</em></p>
    <p style='margin: 5px; font-size: 12px; color: #999;'>Educational tool - Not for medical diagnosis or treatment</p>
</div>
""", unsafe_allow_html=True)