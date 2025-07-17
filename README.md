# Mental Health Counselor Guidance POC

A proof-of-concept web app to help mental health counselors get quick, data-grounded guidance.  
Built with Streamlit, SQLite, a simple ML classifier, and OpenAI’s GPT-3.5 (or a local model).

---

## 📁 Project Structure

mental-health-poc/
├── .env.example # Example environment variables
├── README.md # This file
├── requirements.txt # Python dependencies
├── data/
│ ├── raw/
│ │ └── train.csv # Kaggle “NLP Mental Health Conversations” CSV
│ └── processed/
│ ├── conversations.db # SQLite database of cleaned dialogs
│ ├── conversations.json # JSON export of dialogs
│ └── sample_100.csv # 100-row sample for frontend testing
│
├── notebooks/
│ └── 1-eda.ipynb # Exploratory Data Analysis
│
├── src/
│ ├── load_data.py # Ingest & clean CSV → SQLite + JSON
│ ├── train_model.py # Train TF-IDF + LogisticRegression classifier
│ ├── model_utils.py # predict_advice_type() helper
│ ├── retrieval.py # Semantic search (Sentence-Transformers)
│ ├── llm_client.py # LLM wrapper with few-shot examples
│ └── data_utils.py # (If using SQL-LIKE retrieval instead)
│
├── model/ # Trained artifacts
│ ├── vectorizer.joblib
│ └── advice_clf.joblib
│
└── app.py # Streamlit application entrypoint


---

## ⚙️ Setup

1. **Clone this repo**  
   ```bash
   git clone <your-repo-url> mental-health-poc
   cd mental-health-poc
   
2. **Create & activate a virtual environment (recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\Activate.ps1      # Windows PowerShell
   
3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt

4. **Configure your API key, Make a .env file**  
   ```bash
   OPENAI_API_KEY=sk-YOUR_KEY_HERE

## 🗄️ Data Ingestion

Populate the SQLite database and JSON export:

python src/load_data.py   

Reads data/raw/train.csv

Cleans & drops nulls

Writes data/processed/conversations.db and data/processed/conversations.json


## 📊 Exploratory Data Analysis
jupyter lab notebooks/1-eda.ipynb

Highlights:

Distribution of context & response lengths

Top keywords in patient vs. counselor text

Sample dialogue snippets

Thematic term counts (“anxiety”, “sleep”, etc.)

## 🤖 Train the ML Classifier

python src/train_model.py

Outputs saved to model/:

vectorizer.joblib

advice_clf.joblib

Use src/model_utils.predict_advice_type(text) to classify new responses.

## 🚀 Run the Web App

python -m streamlit run app.py

Then open your browser to: http://localhost:8501




