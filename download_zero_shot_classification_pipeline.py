import os
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

def make_classification_model_directory(model_path, CLASSIFIER_MODEL_PATH):
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(CLASSIFIER_MODEL_PATH):
        os.mkdir(CLASSIFIER_MODEL_PATH)
        print('Made classifier directory')

# zero shot classification pipeline
def download_zero_shot_classifier_pipeline(path):
    tokenizer = AutoTokenizer.from_pretrained("typeform/distilbert-base-uncased-mnli")
    config = AutoConfig.from_pretrained('typeform/distilbert-base-uncased-mnli')
    model = AutoModelForSequenceClassification.from_pretrained("typeform/distilbert-base-uncased-mnli")
    config.save_pretrained(CLASSIFIER_MODEL_PATH + 'config/')
    tokenizer.save_pretrained(CLASSIFIER_MODEL_PATH + 'tokenizer/')
    model.save_pretrained(CLASSIFIER_MODEL_PATH + 'model/')

    print('downloaded zero shot classifier')


if __name__ == '__main__':
    MODEL_PATH = './models/'
    CLASSIFIER_MODEL_PATH = './models/classification/'
    make_classification_model_directory(MODEL_PATH, CLASSIFIER_MODEL_PATH)
    download_zero_shot_classifier_pipeline(CLASSIFIER_MODEL_PATH)
    print('Classification model downloads complete')