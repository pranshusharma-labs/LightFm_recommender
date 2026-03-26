"""
ui.py
─────
Streamlit UI for LightFM recommender.

Usage:
    streamlit run ui.py

Make sure API is running first:
    uvicorn api:app --port 8000
"""

import streamlit as st
import requests
import pandas as pd
import pickle

API_URL   = "http://localhost:8000"
MODEL_DIR = "model/"

st.set_page_config(page_title="LightFM Recommender", page_icon="🛍️", layout="wide")


@st.cache_resource
def load_artifacts():
    try:
        with open(f'{MODEL_DIR}/artifacts.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


artifacts = load_artifacts()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("LightFM Recommender")
    st.markdown("---")

    st.subheader("Model Info")
    if artifacts:
        st.metric("Total Items", len(artifacts['item_id_map']))
        st.metric("Known Users", len(artifacts['user_id_map']))
    else:
        st.warning("Model not found. Run train.py first.")

    st.markdown("---")
    st.subheader("Mode")
    mode = st.radio(
        "Recommendation mode",
        ["hybrid", "collab", "content"],
        format_func=lambda x: {
            "hybrid":  "Hybrid (ID + Features)",
            "collab":  "Pure Collaborative",
            "content": "Pure Content-Based"
        }[x]
    )

    st.markdown("---")
    top_k = st.slider("Top K", 1, 20, 10)


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("LightFM Product Recommender")
st.caption("Collaborative · Content-Based · Hybrid — powered by LightFM")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Recommend", "Eval Results", "Catalog"])

# ── Tab 1: Recommend ──────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")
        user_id = st.text_input(
            "User ID",
            placeholder="e.g. 2258014668",
            help="Must be a known user from training data"
        )
        go = st.button("Get Recommendations", type="primary", use_container_width=True)

    with col2:
        st.subheader("Results")
        if go:
            if not user_id:
                st.warning("Enter a User ID")
            else:
                try:
                    resp = requests.post(
                        f"{API_URL}/recommend",
                        json={"user_id": user_id, "top_k": top_k, "mode": mode},
                        timeout=30
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"Mode: **{data['mode']}** | Items scored: {data['total_items']}")
                        df = pd.DataFrame(data['recommendations'])
                        st.dataframe(
                            df[['rank', 'title', 'item_key', 'score']],
                            use_container_width=True, hide_index=True
                        )
                    elif resp.status_code == 404:
                        st.error(f"User not found in training data.")
                    else:
                        st.error(f"API error {resp.status_code}: {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Run: `uvicorn api:app --port 8000`")

# ── Tab 2: Eval ───────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Offline Evaluation Results")

    if artifacts and 'eval_results' in artifacts:
        ev = artifacts['eval_results']

        rows = []
        for label, key in [
            ("Random Baseline",    "random"),
            ("Pure Collaborative", "collab"),
            ("Pure Content",       "content"),
            ("Hybrid",             "hybrid"),
        ]:
            res = ev.get(key, {})
            for k in [1, 5, 10]:
                if k in res:
                    rows.append({
                        "Model":       label,
                        "K":           k,
                        "Recall@K":    res[k].get('recall',    '-'),
                        "Precision@K": res[k].get('precision', '-'),
                        "NDCG@K":      res[k].get('ndcg',      '-'),
                    })

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption("NDCG@K rewards finding relevant items at higher ranks.")
    else:
        st.info("Run train.py to generate evaluation results.")

# ── Tab 3: Catalog ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Product Catalog")
    try:
        resp = requests.get(f"{API_URL}/items?limit=200", timeout=10)
        if resp.status_code == 200:
            data  = resp.json()
            items = data['items']
            st.caption(f"Showing {len(items)} of {data['total']} items")
            search = st.text_input("Search", placeholder="hoodie, mug...")
            if search:
                items = [i for i in items
                         if search.lower() in i['title'].lower()
                         or search.lower() in i['item_key'].lower()]
            st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
        else:
            st.error("Could not load catalog from API.")
    except requests.exceptions.ConnectionError:
        if artifacts:
            items = [{"item_key": k, "title": artifacts['item_to_title'].get(k, k)}
                     for k in artifacts['item_id_map'].keys()]
            search = st.text_input("Search", placeholder="hoodie, mug...")
            if search:
                items = [i for i in items if search.lower() in i['title'].lower()]
            st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
        else:
            st.error("API not running and no local model found.")
