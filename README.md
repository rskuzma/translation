# Machine Translation Demo
Richard Kuzma, APR 2022<br>
Built with Streamlit front-end and HuggingFace NLP models

<br><br>
### Demonstration purpose:
1) Showcase core NLP use cases for customers:
    - Machine translation
    - Topic Classification
    - Entity Extraction
2) Show peripheral use cases:
    - Translation in multiple languages possible
    - Topics can be added
    - 
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
- Loading models takes ~5 seconds
- Inference takes ~3.5 seconds
- CAVEAT: Running locally on a Macbook Air
    - Running on GPU, distilling models, moving to onyx, quantizing models may all improve performance



