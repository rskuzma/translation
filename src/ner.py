import spacy
import re

spacy.cli.download("en_core_web_sm")
spacy_ner = spacy.load('en_core_web_sm')

def extract_entitites_to_doc(input_text:str, ner_model=spacy_ner):
    spacy_doc = ner_model(input_text)
    return spacy_doc


def entities_to_dict(spacy_doc):
    d = {
        'PERSON':      [],
        'NORP':        [], #nationalities, religious, political groups
        'FAC':         [], # roads, bridges,
        'ORG':         [],
        'GPE':         [], # geopolitical locations (cities, countries)
        'LOC':         [], # non geopolitical locations
        'PRODUCT':     [],
        'EVENT':       [],
        'WORK_OF_ART': [],
        'LAW':         [],
        'LANGUAGE':    [],
        'DATE':        [],
        'TIME':        [],
        'PERCENT':     [],
        'MONEY':       [],
        'QUANTITY':    [],
        'ORDINAL':     [],
        'CARDINAL':    []
        }

    for ent in spacy_doc.ents:
        if ent.label_ in d.keys():
            d[ent.label_].append(ent.text)

    d_slim = {k: v for k, v in d.items() if v}
    return d_slim