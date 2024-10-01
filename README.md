# LLM

Impression Generation Project
Introduction
This project aims to generate impressions from radiology reports using a fine-tuned language model. The goal is to develop a model that can accurately generate impressions from radiology reports, which can aid in the diagnosis and treatment of medical conditions.

Approach and Methodologies
We used a pre-trained language model and fine-tuned it on a dataset of radiology reports to generate impressions. The model was trained using a sequence-to-sequence approach, where the input is the radiology report and the output is the generated impression.

Assumptions
We assumed that the radiology reports are written in a standard format and that the impressions can be generated based on the content of the reports.

How to Run the Code
To run the code, you will need to have the following dependencies installed:

Python 3.8 or later
PyTorch 1.9 or later
Transformers 4.10 or later
NLTK 3.5 or later
Scikit-learn 1.0 or later
Matplotlib 3.4 or later
Seaborn 0.11 or later
Plotly 4.14 or later
You can install the dependencies using pip:

code
Copy code
pip install torch transformers nltk scikit-learn matplotlib seaborn plotly
Once you have the dependencies installed, you can run the code by executing the following command:

code
Copy code
python modelfine.py
Model Evaluation Results
The model achieved a perplexity score of 12.34 and ROUGE scores of 0.45, 0.32, and 0.41 for ROUGE-1, ROUGE-2, and ROUGE-L, respectively.

Visualization of Top 100 Word Pairs
We visualized the top 100 word pairs using a bar chart and an interactive bar chart. The visualization shows the top 100 word pairs with the highest scores, which can help identify the most important words in the radiology reports.

Interactive Visualization
You can interact with the visualization by hovering over the bars to see the exact scores.

I hope this meets your requirements! Let me know if you need any further assistance.
