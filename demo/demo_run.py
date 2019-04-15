import re
import json

import keras
from keras.models import model_from_json
from flask import Flask
from flask import render_template
from flask import request

import pandas as pd

from utils import process_input

data_path = "../../data/emotion.data"
model_json_file = "../notebooks/model.json"
model_weight = "../notebooks/my_model_weights.h5"

max_words = 0
word2id = {}
label2id = {}
id2label = {}

model = None
model_with_attentions = None

app = Flask(__name__)

@app.route('/')
def index():
    assert max_words==178
    assert len(word2id)!=0
    assert len(label2id)!=0
    assert len(id2label)!=0
    return render_template('index.html')

@app.route('/response', methods=['post'])
def response():
    print("RESPONSE ...")
    keras.backend.clear_session()
    
    # assert model is not None
    # assert model_with_attentions is not None
    # print('*Model is {}'.format(model))
    
    with open(model_json_file, 'r') as load_f:
        load_dict = json.load(load_f)
    model = model_from_json(load_dict)
    model.load_weights(model_weight)
    model_with_attentions = keras.Model(inputs=model.input, outputs=[model.output, model.get_layer('attention_vec').output])
    
    input_text = request.form.get('text')
    print(input_text)
    # TODO empty detection; bufer display
    input_text = input_text.strip()
    input_text = re.split("\s*", input_text)
    tokenized_sample = process_input(input_text)
    encoded_samples = [[word2id[word] for word in tokenized_sample]]
    encoded_samples = keras.preprocessing.sequence.pad_sequences(encoded_samples, maxlen=max_words)
    label_probs, attentions = model_with_attentions.predict(encoded_samples)
    label_probs = {id2label[_id]: prob for (label, _id), prob in zip(label2id.items(), label_probs[0])}
    print(label_probs)
    print(attentions[0])
    result = []
    for label in label_probs:
        if label_probs[label] > 0.5:
            result.append(label)

    return render_template('response.html', result=result)

if __name__ == '__main__':
    print("Starting...")
    
    dataset = pd.read_csv(data_path, engine='python')
    input_sentences = [text.split(" ") for text in dataset["text"].values.tolist()]
    labels = dataset["emotions"].values.tolist()
    # word2id = dict()
    # max_words = 0
    for sentence in input_sentences:
        for word in sentence:
            if word not in word2id:
                word2id[word] = len(word2id)
        if len(sentence) > max_words:
            max_words = len(sentence)
    label2id = {l: i for i, l in enumerate(set(labels))} #
    id2label = {v: k for k, v in label2id.items()}       #
    
    app.run(host='0.0.0.0', debug=False)
