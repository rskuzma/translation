from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

def make_translation_model_directory(MODEL_PATH:str, TRANSLATE_MODEL_PATH:str):
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
        if not os.path.exists(TRANSLATE_MODEL_PATH):
            os.mkdir(TRANSLATE_MODEL_PATH)
            print('Made translation model directory')

def download_arabic(TRANSLATE_MODEL_PATH: str):
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    tokenizer.save_pretrained(TRANSLATE_MODEL_PATH + 'arabic/')
    model.save_pretrained(TRANSLATE_MODEL_PATH + 'arabic/')
    print('Saved translation model: arabic')

def download_russian(TRANSLATE_MODEL_PATH: str):
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    tokenizer.save_pretrained(TRANSLATE_MODEL_PATH + 'russian/')
    model.save_pretrained(TRANSLATE_MODEL_PATH + 'russian/')
    print('Saved translation model: russian')

def download_french(TRANSLATE_MODEL_PATH: str):
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    tokenizer.save_pretrained(TRANSLATE_MODEL_PATH + 'french/')
    model.save_pretrained(TRANSLATE_MODEL_PATH + 'french/')
    print('Saved translation model: french')

if __name__ == '__main__':
    MODEL_PATH = './models/'
    TRANSLATE_MODEL_PATH = './models/translation/'
    make_translation_model_directory(MODEL_PATH, TRANSLATE_MODEL_PATH)
    download_arabic(TRANSLATE_MODEL_PATH)
    download_russian(TRANSLATE_MODEL_PATH)
    download_french(TRANSLATE_MODEL_PATH)
    print('Translation model downloads complete')