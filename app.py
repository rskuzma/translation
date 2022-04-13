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
    if language == 'arabic':
        tokenizer, model = translate.arabic()
    elif language == 'russian':
        tokenizer, model = translate.russian()
    elif language == 'french':
        tokenizer, model = translate.french()
    # st.write(f'loaded model: {language}')
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
    if time_it:
        t_start = time.time()
    st.subheader('Step 1: Translating...')
    tokenizer, model = load_hf_models(input_language)
    st.write('Input text: ' + input_text)
    english_text = translate.translate_one_line(input_text, model, tokenizer)
    st.write('English translation: ' + english_text)
    st.text('\n\n')

    if input_classify == 'Yes':
        st.subheader('Step 2: Classifying...')
        st.write('Loading classification model...')
        zero_shot_classifier = classify.load_zero_shot_classifier()
        st.text('\n\n')


        classification_dict_raw = classify.multi_label_classify(input_text=english_text, 
                                                            potential_labels=topics,
                                                            classifier=zero_shot_classifier)
        classification_dict_slim = {k: v for k, v in classification_dict_raw.items() if k in ['labels', 'scores']}
        
        st.table(data=classification_dict_slim)
        # st.write(classification_dict)

    if input_ner == 'Yes':
        st.subheader('Step 3: Extracting Entities...')
        doc = ner.extract_entitites_to_doc(english_text)
        entity_dict = ner.entities_to_dict(doc)
        st.write(entity_dict)
        st.text('\n\n')
    
    t_end = time.time()
    total_time = round(t_end-t_start, 2)
    st.write(f'Time: {total_time} seconds')



    


