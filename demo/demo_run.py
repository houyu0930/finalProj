import re
import json
#import argparse

from flask import Flask
from flask import render_template
from flask import request
import pandas as pd

import keras
from keras.models import model_from_json

from utils import process_input

#OPTS = None

model_path = "../../my_model_0221.h5"
data_path = "../../data/emotion.data"
model_json_file = "../notebooks/model.json"
model_weight = "../notebooks/my_model_weights.h5"

'''
def parse_args():
    parser = argparse.ArgumentParser('Start a demo server.')
    parser.add_argument('--model', default=model_path, help='Full path of the model')
    parser.add_argument('--dataset', default=data_path, help='Full path of the dataset')
    parser.add_argument('--hostname', default='0.0.0.0', help='Hostname')
    parser.add_argument('--port', default=5000, help='Port number')

    return parser.parse_args()
'''

with open(model_json_file, 'r') as load_f:
    load_dict = json.load(load_f)

    # print(load_dict)

model = model_from_json(load_dict)
model.load_weights(model_weight)

global model_with_attentions
model_with_attentions = keras.Model(inputs=model.input, outputs=[model.output, model.get_layer('attention_vec').output])

print(model_with_attentions)

# i = 0
# Build ID base
dataset = pd.read_csv(data_path, engine='python')
input_sentences = [text.split(" ") for text in dataset["text"].values.tolist()]
labels = dataset["emotions"].values.tolist()

word2id = dict()
max_words = 0
for sentence in input_sentences:
    for word in sentence:
        if word not in word2id:
            word2id[word] = len(word2id)
    if len(sentence) > max_words:
        max_words = len(sentence)
label2id = {l: i for i, l in enumerate(set(labels))}
id2label = {v: k for k, v in label2id.items()}

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/response', methods=['post'])
def response():
    #nonlocal i
    print("RESPONSE ...")
    input_text = request.form.get('text')
    print(input_text)

    # TODO empty detection

    # save
    input_text = input_text.strip()
    # TODO display = input_text

    input_text = re.split("\s*", input_text)
    tokenized_sample = process_input(input_text)
    encoded_samples = [[word2id[word] for word in tokenized_sample]]
    encoded_samples = keras.preprocessing.sequence.pad_sequences(encoded_samples, maxlen=max_words)

    print(encoded_samples)
    print(model_with_attentions)
    label_probs, attentions = model_with_attentions.predict(encoded_samples)
    print(label_probs)

    label_probs = {id2label[_id]: prob for (label, _id), prob in zip(label2id.items(), label_probs[0])}

    result = []
    for label in label_probs:
        if label_probs[label] > 0.5:
            result.append(label)

    return render_template('response.html', result=result)

if __name__ == '__main__':
    print("Starting...")
    app.run(host='0.0.0.0', debug=False)


