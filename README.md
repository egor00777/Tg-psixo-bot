# Tg-psixo-bot
Аn assistant for classifying possible mental illnesses and a bot assistant for maintaining psychological health
_____________________________________________________________________________________
# What did we do
1) Analysis of existing solutions:
   Here are some links to similar topics or projects:

   a) Acoustic speech markers for schizophrenia-spectrum disorders (https://www.cambridge.org/core/journals/psychological-medicine/article/acoustic-speech-markers-for-schizophreniaspectrum-disorders-a-diagnostic-and-symptomrecognition-tool/CD60278BD0F09390E8987CB5AB8A887F)

   b) Self-monitoring and personalized feedback (https://research.rug.nl/en/publications/self-monitoring-and-personalized-feedback-based-on-the-experienci/datasets/)

   c) Evaluation of ChatGPT for NLP (https://arxiv.org/abs/2303.15727)
   
3) Datasets used:

   a) binnar_rus_mental.csv - Dataset for binary classification, thanks to it the assessment of “ilness” or “not illness” is made
   b) new_data_class.csv - A dataset for multi-class classification, thanks to which one can choose one of 6 classes of severe mental illnesses
   
_______________________________________
# Several features of the project
*Support for voice messages and post-processing
*Works for Saiga-7b-lora and, with the help of prompt engineering, responds like a real psychotherapist
*If he can’t accurately determine whether a person is sick, he produces a funny GIF
