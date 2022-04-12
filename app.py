import streamlit as st
import os
import sys

SRC_PATH = os.path.join(os.getcwd(), 'src')
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

import translate

st.header('Skope Machine Translation Demo')

with st.sidebar:
    input_language = st.radio('Select language', ['arabic', 'russian', 'french'])
    st.write(input_language)

# @st.cache
def load_hf_models(language:str):
    if language == 'arabic':
        tokenizer, model = translate.arabic()
    elif language == 'russian':
        tokenizer, model = translate.russian()
    elif language == 'french':
        tokenizer, model = translate.french()
input_text = st.text_input('Enter text to be translated')

run = st.button(label="Translate")

if run:
    tokenizer, model = load_hf_models(input_language)
    output_text = translate.translate_one_line(input_text, model, tokenizer)
    st.write(output_text)
