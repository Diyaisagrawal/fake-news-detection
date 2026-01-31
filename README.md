# 📰 Fake News Detection using Bi-LSTM

An end-to-end deep learning project that classifies news articles as **Fake** or **Real** using **Natural Language Processing (NLP)** and a **Bidirectional LSTM (Bi-LSTM)** model.  
The system is deployed as a **Streamlit web application** with confidence scoring and explainability using **LIME**.

---

## 🚀 Live Demo
🔗 Deployed App:  
PASTE_YOUR_STREAMLIT_LINK_HERE

---

## 📌 Problem Statement
The rapid spread of misinformation on digital platforms has made fake news detection a critical challenge. Manual verification is time-consuming and unreliable at scale. This project aims to build an automated and explainable fake news detection system using deep learning techniques.

---

## 🎯 Objectives
- Detect fake vs real news articles using deep learning  
- Capture contextual information from text using Bi-LSTM  
- Display prediction confidence scores  
- Provide explainability using LIME  
- Deploy a real-time, interactive web application  

---

## 🧠 Model Architecture
Text Input
↓
Text Cleaning & Tokenization
↓
Embedding Layer
↓
Bidirectional LSTM
↓
Dropout
↓
Dense (Sigmoid)
↓
Fake / Real Prediction + Confidence Score


---

## 📊 Dataset
- Source: Fake and Real News Dataset (Kaggle)
- Language: English
- Labels: Fake (0), Real (1)
- Size: ~40,000 news articles

---

## ⚙️ Tech Stack
- Python  
- TensorFlow / Keras  
- NLTK  
- LIME (Explainable AI)  
- Streamlit  
- Scikit-learn  
- Matplotlib  

---

## 🔍 Key Features
- Real-time fake news classification  
- Confidence score for each prediction  
- LIME-based word-level explanations  
- Simple and clean web interface  
- Publicly deployable application  

---

## 🖥️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detection.git
cd fake-news-detection
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the application
```bash
streamlit run app.py
```
## 📈 Results
- Achieved approximately **90%+ accuracy** on unseen test data  
- Bi-LSTM effectively captures contextual dependencies in news text  
- LIME highlights important words influencing model predictions  

---

## 🧪 Sample Output
- **Prediction:** REAL  
- **Confidence:** 92.3%  
- **Explanation:** Key words contributing to the prediction are highlighted using LIME  

---

## 🚧 Limitations
- Works only for **English-language** news articles  
- Uses only **textual content** (does not consider source or author credibility)  
- LIME explanations are **local** and may vary across different inputs  

---

## 🔮 Future Enhancements
- Support for **article URLs** with automatic text extraction  
- **Multilingual** fake news detection  
- Integration of **Transformer-based models** (BERT / DistilBERT)  
- **Multi-class classification** (Fake, Satire, Clickbait)  

---

## 👤 Author
**Diya Agrawal**

---

## ⭐ Acknowledgements
- Kaggle Fake and Real News Dataset  
- Streamlit Community  
- LIME Explainable AI Framework  
