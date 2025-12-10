📖 Overview
Clinical notes are essential for patient care but often contain technical language that limits patient comprehension. This application automatically condenses complex biomedical text into concise, accessible summaries using a hybrid NLP approach.
🎯 Key Features

👤 Patient Mode: Upload or paste medical notes and receive easy-to-understand summaries
⚕️ Healthcare Provider Mode: Contribute medical terms and their simple explanations to improve the dictionary
🤖 AI-Powered: Hybrid LexRank + BioBART model achieving 0.40 ROUGE-1 score
📖 Readable: Summaries achieve 33.5 Flesch readability (college level - patient accessible)
📚 Growing Dictionary: Community-driven medical terminology database with 70+ terms
🔄 Real-time Processing: Instant summarization and term simplification


🚀 Live Demo
Try it now: [clinicalsummarization.streamlit.app](https://clinicalsummarization.streamlit.app/)
Sample Usage
For Patients:

Select "I'm a Patient"
Upload a medical note or paste text
Click "Simplify My Notes"
Get an easy-to-understand summary with readability score

For Healthcare Providers:

Select "I'm a Healthcare Provider"
Use "Simplify Notes" tab to test the system
Use "Manage Dictionary" tab to add new medical terms and their simple explanations
View "Dictionary Stats" to see the growing knowledge base


🏗️ Architecture
Hybrid Summarization Pipeline
Medical Note (3000+ words)
         ↓
   LexRank Extraction
   (Select top-15 key sentences)
         ↓
  Condensed Text (~500 words)
         ↓
   BioBART Rewriting
   (Generate fluent summary)
         ↓
   Term Simplification
   (Replace medical jargon)
         ↓
  Patient-Friendly Summary (~200 words)
Technology Stack

NLP Models:

LexRank (extractive baseline)
BioBART (GanjinZero/biobart-base) for abstractive generation


Framework: Streamlit for web interface
Libraries: PyTorch, HuggingFace Transformers, Sumy, NLTK
Storage: JSON-based medical dictionary with persistent storage


📊 Performance Results
Evaluated on 100 PubMed test samples:
MetricScoreROUGE-10.3997ROUGE-20.1328ROUGE-L0.2047Flesch Readability33.5 (College Level)
Comparison to Baselines:

10% improvement over pure extractive methods
5% improvement over BioBART alone
27% better readability than BioBART


💻 Local Installation
Prerequisites

Python 3.8 or higher
pip package manager
4GB+ RAM (for BioBART model)

Setup Instructions

Clone the repository

bashgit clone https://github.com/hemalikaa/Clinical_Summarization_IE7500.git
cd Clinical_Summarization_IE7500/streamlit_app

Install dependencies

bashpip install -r requirements.txt

Download NLTK data (one-time setup)

bashpython -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

Run the application

bashstreamlit run app.py

Open in browser
The app will automatically open at http://localhost:8501


📁 Project Structure
Clinical_Summarization_IE7500/
├── streamlit_app/
│   ├── app.py                      # Main Streamlit application
│   ├── requirements.txt            # Python dependencies
│   ├── medical_dictionary.json     # Medical terminology database
│   ├── packages.txt                # System packages for deployment
│   └── README.md                   # This file
│
└── week9_results/
    ├── final_comprehensive_comparison.png
    ├── WEEK9_FINAL_RESULTS.csv
    ├── HYBRID_ALL_SUMMARIES_100.csv
    └── IE_7500.ipynb               # Research notebook

🔧 How It Works
For Patients

Upload or paste your medical notes
AI extracts key information using LexRank
BioBART generates a fluent, coherent summary
Medical terms are automatically replaced with simple explanations
Readability score indicates how easy the text is to understand

For Healthcare Providers

Test the summarizer with clinical notes
Add new medical terms when you encounter jargon not in the dictionary
Provide simple explanations that patients can understand
Contribute to building a comprehensive medical dictionary

Medical Dictionary
The app maintains a growing JSON database of medical terms and their patient-friendly explanations:
json{
  "myocardial infarction": "heart attack",
  "dyspnea": "shortness of breath",
  "hypertension": "high blood pressure",
  "tachycardia": "fast heart rate"
}
Providers can add new terms through the web interface, and changes are immediately available for all users.

📚 Research Background
This application is based on comprehensive research comparing six summarization approaches:

LexRank (extractive baseline)
T5-Small (generic transformer)
T5-Small Optimized (tuned parameters)
SciFive (scientific T5 variant)
BioBART (biomedical BART)
Hybrid (LexRank + BioBART) ⭐ Winner

Key Findings:

Domain-specific pretraining (BioBART) significantly benefits medical summarization
Hybrid approaches outperform pure extractive or abstractive methods
Combining content selection with fluent rewriting balances accuracy and readability

Dataset: PubMed Summarization Dataset (119,924 training articles, mean length 3,081 words)

🎓 Academic Context
Course: IE 7500 - Neural Natural Language Processing

Institution: Northeastern University
Semester: Fall 2024
Project Goal: Develop NLP systems to automatically condense complex biomedical text into patient-accessible summaries

⚠️ Disclaimer
This is an educational tool for demonstration purposes.
This application:

✅ Simplifies medical language for better understanding
✅ Provides readability assessments
✅ Helps patients prepare questions for their doctors

This application does NOT:

❌ Provide medical diagnosis
❌ Offer treatment recommendations
❌ Replace consultation with healthcare professionals

Always consult your doctor or healthcare provider for medical advice, diagnosis, or treatment.

🤝 Contributing
We welcome contributions! Here's how you can help:
For Healthcare Providers

Add medical terms and explanations through the app interface
Test with real clinical notes and provide feedback
Report terms that need better explanations

For Developers

Submit bug reports via GitHub Issues
Propose new features or improvements
Contribute code via Pull Requests

 Acknowledgments

Qurat-ul-Ain Azim at Northeastern University
PubMed Central for providing the open-access dataset
HuggingFace for transformer model implementations
Streamlit for the web framework
GanjinZero for the BioBART model
