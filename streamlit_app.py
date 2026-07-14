"""Optional Streamlit dashboard for running the AutoML workflow."""

from __future__ import annotations

import pandas as pd

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit("Streamlit is not installed. Install requirements to use the dashboard.") from exc


st.set_page_config(page_title="AutoML Framework", layout="wide")
st.title("AutoML Framework")
st.write("Upload a CSV dataset and inspect the basic structure.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file, na_values=["?"], skipinitialspace=True)
    st.subheader("Preview")
    st.dataframe(dataframe.head())
    st.subheader("Shape")
    st.write(dataframe.shape)
    st.subheader("Missing Values")
    st.dataframe(dataframe.isna().sum().reset_index().rename(columns={"index": "Feature", 0: "Missing"}))
    st.info("Run the full training workflow from main.py after placing the dataset in datasets/adult.csv.")
