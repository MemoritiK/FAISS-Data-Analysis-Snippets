# FAISS Code Retrieval Tool

A **web-based tool for Python developers** to quickly search, explore, and understand data analysis code snippets. Perfect for situations when you’re stuck, confused, or need a quick reference. Combines **FAISS semantic search** with **LLM-powered explanations** for instant insights.

## Features

* **Fast Code Search**:
  Search 800+ Python snippets using keywords, partial code, or natural language queries.

* **Data Analysis Library Support**:
  Works with `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, and more.

* **AI Explanations**:
  Click a snippet to get clear, concise AI-generated explanations using kat-coder-pro.

* **Copy-to-Clipboard**:
  Easily copy code snippets for reuse in your own projects.

* **Category & Context Display**:
  Each snippet shows its core category and an associated question for context.

* **Run Snippets**:
  Execute code snippets directly in a sandboxed environment with inline output.

* **Clean, Minimal UI**:
  Streamlit-based frontend for quick access and smooth workflow.

## How It Works

1. **Search**:
   Enter a query → frontend sends a POST request to `/search`.

2. **Retrieve**:
   Backend searches preprocessed snippets using FAISS → returns top matches.

3. **Display**:
   Frontend shows snippet code, category, and associated question.

4. **Explain**:
   Click “Explain” → frontend calls `/explain` → AI explanation displayed inline.

5. **Copy**:
   Click “Copy” → snippet code copied to clipboard.

6. **Execution**:
   Click “Run” → backend executes code safely → output or plots shown inline.

## Live Demo

* **Frontend:** [https://faiss-data-analysis-snippets-9qgw4qsfc5jviqqkckwikv.streamlit.app/](https://faiss-data-analysis-snippets-9qgw4qsfc5jviqqkckwikv.streamlit.app/)
* **Backend API:** [https://faiss-data-analysis-snippets.onrender.com](https://faiss-data-analysis-snippets.onrender.com)

## Tech Stack

* **Backend:** Python, Flask, FAISS, NumPy, pickle, ONNX Runtime
* **Frontend:** Streamlit
* **AI:** kwaipilot/kat-coder-pro:free from OpenRouter

## Installation

```bash
git clone <repo_url>
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Start the Flask backend from /backend:

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
Flask
flask-cors
numpy
pandas
matplotlib
seaborn
scipy
faiss-cpu
requests
pyyaml
tqdm
onnxruntime
statistics
random
pathlib
json
```
## Results
<img width="560" height="907" alt="image" src="https://github.com/user-attachments/assets/3257e1bf-1d55-475b-baa7-6cd03281f8c0" />
<img width="537" height="908" alt="image" src="https://github.com/user-attachments/assets/0a7996d2-187f-4a37-bdc0-8b8e92049622" />
<img width="583" height="910" alt="image" src="https://github.com/user-attachments/assets/a3bda74f-09b5-4fe5-8b72-31dd48b3f234" />

