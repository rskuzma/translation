from transformers import pipeline

def load_zero_shot_classifier(model='typeform/distilbert-base-uncased-mnli'):
    return pipeline("zero-shot-classification", model=model)

def multi_label_classify(input_text:str,
                         potential_labels:list,
                         classifier,
                         multi_label=True
                        ):
    output_dict = classifier(input_text, potential_labels, multi_label=multi_label)
    return output_dict