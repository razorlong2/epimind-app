#!/usr/bin/env python3
"""
Script pentru instalarea modelelor spaCy pe Streamlit Cloud
Rulează automat după instalarea pachetelor
"""

import subprocess
import sys
import os

def install_spacy_models():
    """Instalează modelele spaCy necesare"""
    models = [
        'ro_core_news_sm',  # Model pentru română
        'en_core_web_sm'    # Model pentru engleză
    ]
    
    for model in models:
        try:
            print(f"📥 Instalând modelul spaCy: {model}")
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", model
            ])
            print(f"✅ {model} instalat cu succes")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Nu s-a putut instala {model}: {e}")
        except Exception as e:
            print(f"❌ Eroare la instalarea {model}: {e}")

def setup_nltk_data():
    """Descarcă datele NLTK necesare"""
    try:
        import nltk
        print("📚 Descărcare date NLTK...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ Date NLTK descărcate")
    except Exception as e:
        print(f"⚠️ Nu s-au putut descărca datele NLTK: {e}")

def main():
    print("🚀 Post-instalare EpiMind pentru Streamlit Cloud")
    print("=" * 50)
    
    if os.getenv('STREAMLIT_SHARING_MODE') or os.getenv('STREAMLIT_CLOUD'):
        print("☁️ Detectat mediu Streamlit Cloud")
    else:
        print("💻 Mediu de dezvoltare local")
    
    install_spacy_models()
    setup_nltk_data()
    
    print("\n✅ Post-instalare completă!")

if __name__ == "__main__":
    main()