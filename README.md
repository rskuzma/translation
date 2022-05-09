# Machine Translation Demo
Richard Kuzma, MAY 2022<br>
Built with Streamlit front-end and HuggingFace NLP models

#### How to run:
1) Clone git
    a) Make sure large files (pytorch models) are cloned too
    b) These are saved here for local running (No internet)
    c) Two files start with `download*` that are helpful for downloading models for first time
2) `cd` into repo
3) Build environment with requirements.txt
4) In terminal, run the command: `streamlit run app.py`



<br><br>
### Demonstration purpose:
1) Showcase core NLP use cases for customers:
    - Machine translation
    - Topic Classification
    - Entity Extraction
2) Show peripheral use cases:
    - Translation in multiple languages possible
    - Topics can be added
3) Streamlit as a low-lift front-end
    - Python wrapper around React front-end components
    - Easy for Python programmers to use
    - Better user experience than dealing with jupyter noteboks
        - Easy to change inputs via buttons, text, drop-downs

<br><br>
### Caveats:
1) Off-the-shelf models, not fine-tuned. Human in the loop required for assessing validity

<br><br>
### Time to run:
- Loading models takes ~4 seconds
- Inference takes ~2.5 seconds
- CAVEAT: Running locally on a Macbook Air
    - Models saved locally
    - Running on GPU, distilling models, moving to onyx, quantizing models may all improve performance
    - Horizontal scaling needed for more throughput?



