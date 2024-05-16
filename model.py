import keras
import pandas as pd
import torch
from huggingsound import SpeechRecognitionModel
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, \
    Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, pipeline, WhisperProcessor, WhisperForConditionalGeneration


class BinarModel:
    def __init__(self):
        self.df = pd.read_csv('binar_mental_clear.csv')
        self.X = self.df['text1']
        self.y = self.df['label']

        self.MAX_FEATURES = 200000

        self.vectorizer = keras.layers.TextVectorization(
            max_tokens=self.MAX_FEATURES,
            output_sequence_length=1000,
            output_mode='int'
        )

        self.vectorizer.adapt(self.X.values)
        self.vectorizerd_text = self.vectorizer(self.X.values)
        self.model_loaded = keras.models.load_model('binar_model_en_new_LSTM.keras')

    def predict(self, input_text):
        input_text = self.vectorizer([input_text])
        pred = self.model_loaded.predict(input_text)
        return pred[0][0]


class ClassModel:
    def __init__(self):
        self.df = pd.read_csv('new_data_class.csv')
        self.X = self.df['text1']
        self.y = self.df['res']

        self.MAX_FEATURES = 200000

        self.vectorizer = keras.layers.TextVectorization(
            max_tokens=self.MAX_FEATURES,
            output_sequence_length=1000,
            output_mode='int'
        )

        self.vectorizer.adapt(self.X.values)
        self.vectorizerd_text = self.vectorizer(self.X.values)
        self.model_loaded = keras.models.load_model('model_classifier_en_new_LSTM.keras')

    def predict(self, input_text):
        input_text = self.vectorizer([input_text])
        pred = self.model_loaded.predict(input_text)
        # print(pred)
        return pred[0]


class SageFredT5:
    def __init__(self, model_path="ai-forever/sage-fredt5-distilled-95m"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model = self.model.to("cuda") if torch.cuda.is_available else self.model

    def correct_text(self, sentence):
        inputs = self.tokenizer(sentence, max_length=None, padding="longest", truncation=False, return_tensors="pt")
        outputs = self.model.generate(**inputs.to(self.model.device), max_length=inputs["input_ids"].size(1) * 1.5)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


class SpeechModel:
    def __init__(self):
        self.model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
        self.sage = SageFredT5()

    def recognize(self, path):
        transcriptions = self.model.transcribe([path])
        return self.sage.correct_text(transcriptions[0]['transcription'])
