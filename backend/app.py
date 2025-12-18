from flask import Flask, request, jsonify
from flask_cors import CORS
from code_retrevial.progression_engine import search_snippets
from code_retrevial.llm_inference import get_explanation
from code_retrevial.exec_code import run_python_code
import numpy as np

app = Flask(__name__)
CORS(app)

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


@app.route('/search', methods=['POST'])
def code_query():
    data = request.get_json()
    query = data.get("query")
    top_k = data.get("top_k", 3)
    difficulty = data.get("difficulty", "all")
    result = search_snippets(query, top_k, difficulty)
    result_clean = [
        {k: convert_numpy(v) for k, v in snippet.items()}
        for snippet in result
    ]
    return jsonify(result_clean)

@app.route('/explain', methods=['POST'])
def code_explain():
    data = request.get_json()
    content = data.get("code", "")
    explanation = get_explanation(content)
    return jsonify({"explanation": explanation})
    
@app.post("/run")
def run_code():
    data = request.get_json()
    code = data.get("code", "")

    result = run_python_code(code)
    return jsonify(result)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
