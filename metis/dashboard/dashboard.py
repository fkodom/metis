"""
# Metis Dashboard
"""
import argparse

import streamlit as st


parser = argparse.ArgumentParser()
parser.add_argument("logdir")
logdir = parser.parse_args().logdir

st.sidebar.markdown("# Metis Dashboard")
refresh = st.sidebar.button("Refresh Data")
if refresh:
    st.caching.clear_cache()

st.sidebar.markdown("""### Navigation""")
page = st.sidebar.radio(
    "Section: ",
    ("Metrics", "Hyperparameters", "Computation Graphs", "Histograms")
)

"""
# Metis Dashboard
"""