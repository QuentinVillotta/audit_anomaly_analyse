from pygwalker.api.streamlit import StreamlitRenderer
import pandas as pd
import streamlit as st
import base64
import json
import pickle
import uuid
import re

def set_clicked():
    st.session_state.clicked = True
    
# You should cache your pygwalker renderer, if you don't want your memory to explode
@st.cache_resource
def get_pyg_renderer(df) -> "StreamlitRenderer":
    # If you want to use feature of saving chart config, set `spec_io_mode="rw"`
    return StreamlitRenderer(df, spec="./gw_config.json", spec_io_mode="rw")


# Adjust the width of the Streamlit page
st.set_page_config(
    page_title="Features Explorer",
    layout="wide"
)

# Title
st.title("Features Explorer")


if 'clicked' not in st.session_state:
    st.session_state.clicked = False

st.button('Upload File', on_click=set_clicked)
if st.session_state.clicked:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is  None:
        st.write("Please upload a dataset")
    else:
        st.write("You selected the file:", uploaded_file.name)
        data = pd.read_excel(uploaded_file)
        renderer = get_pyg_renderer(df=data)
        renderer.explorer()