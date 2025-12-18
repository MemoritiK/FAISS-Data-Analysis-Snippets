# FAISS Code Retreival Tool

A **web-based tool for Python developers** to quickly search, explore, and understand data analysis code snippets. Designed for **fast reference** when stuck or exploring new techniques, combining **FAISS semantic search** with **AI-powered explanations**.


## Features

* **Fast Code Search**
  Search 800+ Python snippets using keywords, partial code, or natural language queries.

* **Data Analysis Library Support**
  Works with `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, and more.

* **AI Explanations**
  Click a snippet to get clear, concise AI-generated explanations using DeepSeek LLM.

* **Copy-to-Clipboard**
  Easily copy code snippets for reuse in your own projects.

* **Category & Context Display**
  Each snippet shows its core category and an associated question for context.

* **Planned: Run Snippets**
  Future feature: Execute code snippets directly in a sandboxed environment with inline output.

* **Clean, Minimal UI**
  Streamlit-based frontend for quick access and smooth workflow.


## How It Works

1. **Search**
   Enter a query → frontend sends a POST request to `/search`.

2. **Retrieve**
   Backend searches preprocessed snippets using FAISS → returns top matches.

3. **Display**
   Frontend shows snippet code, category, and associated question.

4. **Explain**
   Click “Explain” → frontend calls `/explain` → AI explanation displayed inline.

5. **Copy**
   Click “Copy” → snippet code copied to clipboard.

6. **Future Execution**
   Click “Run” → backend executes code safely → output or plots shown inline.


## Tech Stack

* **Backend:** Python, Flask, FAISS, NumPy, pickle, ONNX Runtime
* **Frontend:** Streamlit
* **AI:** DeepSeek LLM for explanations


## Installation

```bash
git clone <repo_url>
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Usage

1. Start the Flask backend:

```bash
python app.py
```

2. Start the Streamlit frontend:

```bash
streamlit run frontend.py
```

3. Open the browser at the Streamlit URL and start searching code snippets.


## Environment Variables

Set your API key for the LLM:

```bash
export API_KEY="your_api_key_here"
```

In your requests:

```python
headers = {"Authorization": f"Bearer {API_KEY}"}
```


## Requirements

```text
Flask==2.3.4
flask-cors==3.0.10
numpy==1.27.5
pandas==2.1.2
matplotlib==3.8.1
seaborn==0.12.3
scipy==1.11.1
faiss-cpu==1.7.4
requests==2.31.0
pyyaml==6.0
tqdm==4.66.1
onnxruntime==1.15.1
statistics
random
pathlib
json
```


