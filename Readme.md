# 🏥 EpiMind - IAAM Predictor cu OCR-NLP

**Platformă demonstrativă pentru evaluarea predictivă a infecțiilor asociate asistenței medicale (IAAM) cu inteligență artificială**

## 🚀 Funcționalități

### Core Features
- ✅ **Predictor IAAM** - Evaluare risc bazată pe algoritmi clinici
- ✅ **Scoruri validate** - SOFA, qSOFA, APACHE-like
- ✅ **Analiză comorbidități** - Catalog structurat afecțiuni
- ✅ **Microbiologie** - Profile de rezistență și patogeni MDR

### 🆕 OCR-NLP Features (v2.3.0)
- 🔬 **OCR Medical** - Extragere automată text din documente medicale
- 🧠 **NLP Avansat** - Detectare valori medicale, bacterii, rezistențe
- ⚡ **Auto-completare** - Populare automată formular cu datele extrase
- 🎯 **Suport multilingv** - Română și engleză

## 📊 Demo Live

**🌐 [Accesați aplicația live pe Streamlit Cloud](https://your-app-url.streamlit.app)**

## 🔧 Instalare Locală

### Dependințe
- Python 3.8+
- Tesseract OCR
- Modele spaCy (română/engleză)

### Quick Start
```bash
# Clonare repository
git clone https://github.com/username/epimind-ocr-nlp.git
cd epimind-ocr-nlp

# Creare mediu virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalare dependințe
pip install -r requirements.txt

# Instalare modele spaCy
python postinstall.py

# Rulare aplicație
streamlit run IAAM_PREDICTOR.py
```

### Instalare Tesseract OCR

**Windows:**
```bash
# Descărcare de la: https://github.com/UB-Mannheim/tesseract/wiki
# Adăugare la PATH
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install tesseract-ocr tesseract-ocr-ron
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

## 📖 Utilizare

### 1. Workflow Standard
1. **Date pacient** - Introduceți informații demografice
2. **Dispozitive** - Selectați dispozitive invazive și durata
3. **Parametri clinici** - Valori pentru SOFA/qSOFA
4. **Microbiologie** - Culturi și profile de rezistență
5. **Evaluare** - Calculare scor IAAM și recomandări

### 2. 🆕 Workflow OCR-NLP
1. **Upload document** - Încărcați analize medicale (PNG, JPG, etc.)
2. **Procesare OCR** - Extragere automată text
3. **Analiză NLP** - Detectare valori medicale
4. **Auto-completare** - Populare automată formular
5. **Evaluare** - Scor IAAM cu datele extrase

## 🏗️ Arhitectură

### Core Engine
- **Temporal Criteria** - Validare IAAM (≥48h internare)
- **Device Scoring** - Punctaj dispozitive invazive
- **Severity Scores** - SOFA, qSOFA, APACHE-like
- **Microbiology** - Patogeni și rezistențe MDR
- **Laboratory** - Markeri inflamatori și infecție

### OCR-NLP Pipeline
- **Image Preprocessing** - Filtrare și îmbunătățire imagine
- **OCR Engine** - Tesseract cu configurare medicală
- **Pattern Recognition** - Regex patterns pentru valori medicale
- **Entity Extraction** - spaCy pentru bacterii și medicamente
- **Data Validation** - Cross-reference și consistență

## 📊 Exemple Utilizare

### Input Documents
- 📄 **Analize laborator** - Hemoleucogramă, biochimie
- 🦠 **Rezultate microbiologice** - Culturi, antibiograme  
- 📋 **Fișe de observație** - Parametri vitali
- 🩺 **Rapoarte imagistice** - Cu valori numerice

### Output
- 🎯 **Scor IAAM** - Numeric cu nivel de risc
- 📋 **Recomandări** - Acțiuni clinice concrete
- 📊 **Breakdown detaliat** - Contribuția fiecărei componente
- 📄 **Export** - JSON, CSV pentru integrare

## 🔒 Securitate și Confidențialitate

- ✅ **Procesare locală** - Datele nu părăsesc sistemul
- ✅ **Anonimizare** - Fără date personale în demo
- ✅ **Audit trail** - Înregistrare activități
- ✅ **Validare input** - Protecție contra injecții

## 📚 Documentație Tehnică

### API Integration
```python
# Exemple de integrare
from epimind import calculate_iaam_risk

result = calculate_iaam_risk({
    'ore_spitalizare': 96,
    'cultura_pozitiva': True,
    'bacterie': 'Escherichia coli',
    'analize': {'wbc': 15.2, 'crp': 120}
})
```

### Custom Patterns
```python
# Personalizare pattern-uri OCR
custom_patterns = {
    'marker_local': r'marker_specific[:\s]*(\d+(?:\.\d+)?)',
    'test_local': r'test_name[:\s]*(\d+(?:\.\d+)?)'
}
```

## 🤝 Contribuții

1. **Fork** repository-ul
2. **Creați branch** pentru feature (`git checkout -b feature/amazing-feature`)
3. **Commit** modificările (`git commit -m 'Add amazing feature'`)
4. **Push** pe branch (`git push origin feature/amazing-feature`)
5. **Deschideți Pull Request**

## 📞 Suport

- 📧 **Email**: support@epimind.ro
- 🐛 **Issues**: [GitHub Issues](https://github.com/username/epimind-ocr-nlp/issues)
- 📖 **Wiki**: [Documentație completă](https://github.com/username/epimind-ocr-nlp/wiki)

## 📜 Licență

Acest proiect este licențiat sub **MIT License** - vezi fișierul [LICENSE](LICENSE) pentru detalii.

## 🏆 Autori

- **Dr. Boghian Lucian** - *UMF "Grigore T. Popa" Iași*
- **Echipa EpiMind** - *Dezvoltare și implementare*

## 🙏 Mulțumiri

- **spaCy** - Pentru modelele NLP românești
- **Tesseract** - Pentru engine-ul OCR
- **Streamlit** - Pentru framework-ul web
- **Comunitatea medicală** - Pentru feedback și validare

---

**⚠️ NOTĂ IMPORTANTĂ**: Aceasta este o platformă demonstrativă pentru scop academic și de cercetare. Nu înlocuiește judecata clinică și nu trebuie folosită pentru diagnosticul medical direct fără supervizare specializată.