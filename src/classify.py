from transformers import PreTrainedTokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import ZeroShotClassificationPipeline
from transformers import DistilBertConfig
import os

# from transformers import logging as hf_logging
# hf_logging.set_verbosity_info()


def load_config_from_local():
    loaded_config = AutoConfig.from_pretrained('./models/classification/config/', local_files_only=True)
    # note use of ./ instead of ../ because this is called from app.py in a higher directory
    # could also use DistilBertConfig.from_pretrained() as that is the model type
    return loaded_config

def load_tokenizer_from_local():
    loaded_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path = './models/classification/tokenizer/', local_files_only=True)
    # type is DistilBertTokenizerFast
    return loaded_tokenizer

def load_model_from_local():
    myconfig = load_config_from_local()
    mymodel = AutoModelForSequenceClassification.from_pretrained(
        # type could be DistilBertForSequenceClassification
            pretrained_model_name_or_path = './models/classification/model/',
            config = myconfig,
        )
    return mymodel


def load_zero_shot_classifier(local=True):
    print('method \'load_zero_shot_classifier\' called')
    if not local: 
        model='typeform/distilbert-base-uncased-mnli'
        return pipeline("zero-shot-classification", model=model)
    else:
        mytokenizer = load_tokenizer_from_local()
        mymodel = load_model_from_local()
        zsl_pipe = ZeroShotClassificationPipeline(model=mymodel, tokenizer=mytokenizer)
        return zsl_pipe

def multi_label_classify(input_text:str,
                         potential_labels:list,
                         classifier,
                         multi_label=True
                        ):
    output_dict = classifier(input_text, potential_labels, multi_label=multi_label)
    return output_dict