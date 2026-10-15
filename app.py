import streamlit as st

from search import Searcher


@st.cache_resource
def load():
    return Searcher()


s = load()
st.title("Semantic search")
q = st.text_input("Query")
mode = st.radio("Method", ["bm25", "dense", "hybrid", "hybrid+rerank"], horizontal=True)

if q:
    if mode == "bm25":
        ids = s.bm25_search(q)
    elif mode == "dense":
        ids = s.dense_search(q)
    elif mode == "hybrid":
        ids = s.hybrid_search(q)
    else:
        ids = s.rerank(q, s.hybrid_search(q, 50))
    for i in ids:
        d = s.docs[i]
        st.markdown(f"**{d.get('title', d['id'])}**")
        st.write(d["text"][:300])
