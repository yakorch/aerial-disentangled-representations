import os
import pathlib

import numpy as np
import streamlit as st

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
from disentangled_representations.src.data_processing.aerial_dataset_instances import Hi_UCD_Dataset, A

PAGE_SIZE = 10
embeddings_A_path = pathlib.Path("app/embeddings/A_embeddings.npy")
embeddings_B_path = pathlib.Path("app/embeddings/B_embeddings.npy")


@st.cache_data
def load_dataset():
    return Hi_UCD_Dataset(split="test", read_color=True, shared_transform=A.Compose([], additional_targets={}), unique_transform=A.Compose([]), )


@st.cache_data
def load_embeddings():
    emb_A = np.load(embeddings_A_path).astype("float32")
    emb_B = np.load(embeddings_B_path).astype("float32")
    return emb_A, emb_B


st.title("🛰️ Aerial Image Retrieval")

dataset = load_dataset()
emb_A, emb_B = load_embeddings()
N = len(dataset)
pages = (N + PAGE_SIZE - 1) // PAGE_SIZE

st.sidebar.header("Options")
page_num = st.sidebar.number_input("Page", min_value=1, max_value=pages, value=1, step=1)
start = (page_num - 1) * PAGE_SIZE
end = min(start + PAGE_SIZE, N)
page_indices = list(range(start, end))

st.sidebar.header("Retrieval Settings")
query_idx = st.sidebar.selectbox("Query image index", options=page_indices, index=0)
k = st.sidebar.slider("Number of neighbors (k)", 1, 20, 5)
direction = st.sidebar.selectbox("Retrieval direction", ["A → B", "B → A"])

st.subheader(f"Query Images — Page {page_num} (idx {start}–{end - 1})")
cols_top = st.columns(5)
for i, idx in enumerate(page_indices[:5]):
    sample = dataset[idx]
    img = sample["A"] if isinstance(sample, dict) else sample[0]
    cols_top[i].image(img, caption=f"Idx {idx}", use_container_width=True)

cols_bot = st.columns(5)
for j, idx in enumerate(page_indices[5:]):
    sample = dataset[idx]
    img = sample["A"] if isinstance(sample, dict) else sample[0]
    cols_bot[j].image(img, caption=f"Idx {idx}", use_container_width=True)

sample = dataset[query_idx]
img_A = sample["A"] if isinstance(sample, dict) else sample[0]
img_B = sample["B"] if isinstance(sample, dict) else sample[1]

if direction == "A → B":
    query_img = img_A
    query_emb = emb_A[query_idx]
    gallery_side = "B"
    gallery_emb = emb_B
else:
    query_img = img_B
    query_emb = emb_B[query_idx]
    gallery_side = "A"
    gallery_emb = emb_A

st.subheader("🔎 Selected Query")
st.image(query_img, use_container_width=True)

sims = gallery_emb @ query_emb
topk = np.argpartition(-sims, k - 1)[:k]
topk_sorted = topk[np.argsort(-sims[topk])]

st.subheader("⭐ Nearest Neighbors")
cols = st.columns(k)
for rank, col in enumerate(cols):
    nbr_idx = int(topk_sorted[rank])
    sim_val = float(sims[nbr_idx])
    is_correct = (nbr_idx == query_idx)
    sample_n = dataset[nbr_idx]
    if isinstance(sample_n, dict):
        nbr_img = sample_n[gallery_side]
    else:
        nbr_img = sample_n[0] if gallery_side == "A" else sample_n[1]

    status_emoji = "✅" if is_correct else "❌"
    col.image(nbr_img, caption=f"{status_emoji} Idx: {nbr_idx}\nSim: {sim_val:.3f}", use_container_width=True)
