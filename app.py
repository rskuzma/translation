## sample input
# US Central Command General Michael Kurilla loves the penguins hockey team based in Pittsburgh, Pennsylvania. Last week he purchased tickets for $1000 by calling his friend Richard at 703-867-5309. 


import streamlit as st
import os
import sys
import time
# import transformers

SRC_PATH = os.path.join(os.getcwd(), 'src')
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

import translate
import classify
import ner

######################################################

st.title('Skope Machine Translation Demo')

######################################################

with st.sidebar:
    time_it = st.radio('Time it?', ['No', 'Yes'])
    input_language = st.radio('Select language', ['arabic', 'russian', 'french'], help='What language is the input?')
    input_classify = st.radio('Topic Classification?', ['No', 'Yes'], help='Do you want to know the text topic?')
    input_ner = st.radio('Extract Entities', ['No','Yes'], help='Extract people, places, things')

######################################################

# @st.cache
def load_hf_models(language:str):
    """Load NLP machine translation models from HuggingFace.

    This application uses HuggingFace, a high-level API for open-source NLP models.
    This function calls the `translate` module for actual loading of the models

    Args:
        language (str): Language of text to be translated. Comes from Streamlit button user input

    Returns:
        tokenizer: HuggingFace tokenizer that will turn text into machine-readable tokens
        model: HuggingFace model doing the translations
    """    
    if language == 'arabic':
        tokenizer, model = translate.arabic()
    elif language == 'russian':
        tokenizer, model = translate.russian()
    elif language == 'french':
        tokenizer, model = translate.french()
        
    return tokenizer, model
    

######################################################        

input_text = st.text_input('Enter text to be translated')


######################################################

run = st.button(label="Translate")


######################################################

topics = st.multiselect(label='Pick potential topics', 
                            options=['sports', 'politics', 'technology', 'economics', 
                                     'healthcare', 'news', 'military', 'travel'], 
                            default=None)

######################################################
       


if run:
    ### Translate
    st.subheader('Step 1: Translate...')
    st.write('Loading translation model')
    translate_model_load_start = time.time()
    tokenizer, model = load_hf_models(input_language)
    translate_model_load_time = time.time() - translate_model_load_start 
    
    st.write('Input text: ' + input_text)
    
    st.write('Translating text')
    translate_time_start = time.time()
    english_text = translate.translate_one_line(input_text, model, tokenizer)
    translate_time = time.time() - translate_time_start


    st.write('English translation: ' + english_text)
    st.text('\n\n\n\n\n')


    ### Topic Classify
    if input_classify == 'Yes':
        st.subheader('Step 2: Classify Topic(s)...')
        st.write('\n\n')
        classify_model_load_start = time.time()
        zero_shot_classifier = classify.load_zero_shot_classifier()
        classify_model_load_time = time.time() - classify_model_load_start
        

        classify_time_start = time.time()
        classification_dict_raw = classify.multi_label_classify(input_text=english_text, 
                                                            potential_labels=topics,
                                                            classifier=zero_shot_classifier)
        classification_dict_slim = {k: v for k, v in classification_dict_raw.items() if k in ['labels', 'scores']}
        classify_time = time.time() - classify_time_start
        st.table(data=classification_dict_slim)


    if input_ner == 'Yes':
        st.subheader('Step 3: Extract Entities...')
        ner_time_start = time.time()
        doc = ner.extract_entitites_to_doc(english_text)
        entity_dict = ner.entities_to_dict(doc)
        ner_time = time.time() - ner_time_start
        st.write(entity_dict)
        st.text('\n\n')
    
    if time_it == 'Yes':
        st.subheader('Times:')
        st.write(f'Time to load translation model: {round(translate_model_load_time, 2)} seconds')
        st.write(f'Time to translate: {round(translate_time, 2)} seconds\n')
        st.write(f'Time to load topic classifier model: {round(classify_model_load_time,2)} seconds')
        st.write(f'Time to classify topics: {round(classify_time, 2)} seconds\n')
        st.write(f'Time to extract entities: {round(ner_time, 2)} seconds\n')
        
        st.write(f'Time to load models: {round(translate_model_load_time + classify_model_load_time,2)} seconds')
        st.write(f'Time to make predictions: {round(translate_time + classify_time + ner_time, 2)} seconds')
        



    


