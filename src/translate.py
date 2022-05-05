from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def arabic(local=True, path_to_model_directory:str = ''):
    if local:
        tokenizer = AutoTokenizer.from_pretrained(path_to_model_directory)
        model = AutoModelForSeq2SeqLM.from_pretrained(path_to_model_directory)    
    else:
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    return tokenizer, model

def russian(local=True, path_to_model_directory:str = ''):
    if local:
        tokenizer = AutoTokenizer.from_pretrained(path_to_model_directory)
        model = AutoModelForSeq2SeqLM.from_pretrained(path_to_model_directory)    
    else:
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    return tokenizer, model

def french(local=True, path_to_model_directory:str = ''):
    if local:
        tokenizer = AutoTokenizer.from_pretrained(path_to_model_directory)
        model = AutoModelForSeq2SeqLM.from_pretrained(path_to_model_directory)    
    else:
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    return tokenizer, model

# def chinese():
#     tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
#     model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
#     return tokenizer, model


def translate_one_line(line:str, model, tokenizer):
    if line == '':
        return ''
    else:
        input_ids = tokenizer.encode(line, return_tensors="pt")
        outputs = model.generate(input_ids)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded