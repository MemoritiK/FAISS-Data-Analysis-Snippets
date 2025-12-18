import streamlit as st
import requests
import base64

st.set_page_config(
    page_title="Code Retrieval",
    layout="centered"
)

BACKEND_URL = "http://127.0.0.1:5000"

if "results" not in st.session_state:
    st.session_state.results = []

if "explanations" not in st.session_state:
    st.session_state.explanations = {}


st.title("FAISS Code Retrieval")


with st.container():
    query = st.text_area(
        "Search query",
        height=80,
        placeholder="e.g. pandas groupby missing values"
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        difficulty = st.selectbox(
            "Difficulty",
            ["all", "easy", "medium", "hard"]
        )

    with col2:
        top_k = st.slider(
            "Results",
            min_value=1,
            max_value=10,
            value=3
        )

    search_clicked = st.button("Search")
        


if search_clicked:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching..."):
            res = requests.post(
                f"{BACKEND_URL}/search",
                json={
                    "query": query,
                    "top_k": top_k,
                    "difficulty": difficulty
                }
            )
            st.session_state.results = res.json()
            st.session_state.explanations = {}


if st.session_state.results:
    st.markdown("### Results")

    for idx, snippet in enumerate(st.session_state.results):
        with st.container():

            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption("Category \n\n" f"{snippet['core_category']}")
                with col2:
                    st.caption("Difficulty" "\n\n" f" {snippet['difficulty']}")
                    with col3:
                        st.caption("Relevance Score" "\n\n" f"{snippet['score']:.3f}")

            # Question
            if snippet.get("question"):
                st.markdown(f"**Q:** {snippet['question']}")

            # Code (native copy button included)
            st.code(snippet["code"], language="python")
            
            explain_key = f"explain_{idx}"

            if st.button("Explain", key=explain_key):
                with st.spinner("Generating explanation..."):
                    res = requests.post(
                        f"{BACKEND_URL}/explain",
                        json={"code": snippet["code"]}
                    )

                    # Backend returns JSON: {"explanation": "..."}
                    explanation = res.json().get("explanation", "")
                    st.session_state.explanations[idx] = explanation

            # Explanation display
            if idx in st.session_state.explanations:
                with st.expander("Explanation", expanded=True):
                    st.markdown(st.session_state.explanations[idx])
            
            run_key = f"run_{idx}"
            output_box = st.empty()
            
            if st.button("Run", key=run_key):
                with st.spinner("Running snippet..."):
                    response = requests.post(
                        f"{BACKEND_URL}/run",
                        json={"code": snippet["code"]}
                    )
            
                result = response.json()
            
                with output_box.container():
                    st.markdown("### Output")
            
                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        if result.get("stdout"):
                            st.code(result["stdout"], language="text")
                        if result.get("last_expression"):
                            st.code(result["last_expression"], language="text")

                        for img in result.get("plots", []):
                            st.image(
                                base64.b64decode(img),
                                use_container_width=True
                            )
            
            st.markdown("---")
            

if not st.session_state.results and not query:
    st.caption(
        "Search for a concept, question, or code pattern to retrieve relevant snippets."
    )
