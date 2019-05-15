import torch
import numpy as np

from flask import Flask
from flask import render_template
from flask import request

from trainer import load_embeddings
from lib.tokenizer import preprocessor
from lib.config import MODEL_EC, DEVICE
from lib.data_utils import vectorize
from lib.utils import load_movies, load_music

app = Flask(__name__)
app._static_folder = './static'

labels = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Love', 'Optimism', 'Pessimism', 'Sadness', 'Surprise', 'Trust', 'Neutral']
recommend_id = [1, 4, 2, 3, 4, 4, 4, 3, 5, 6, 4, 4]
model_path = '/home/houyu/learning/FinalProject/out/model/EmotionClassification_0.5900_2019-05-06_00:51.model'
movie_path = '/home/houyu/learning/FinalProject/database/MovieData.csv'
music_path = '/home/houyu/learning/FinalProject/database/MusicData.csv'
model_conf = MODEL_EC
max_length = 85  # 85 train 65 dev 58 test

# 1   1   0   0   0   0   1   0   0   0   1
# test_sentence = '@Adnan__786__ @AsYouNotWish Dont worry Indian army is on its ways to dispatch all Terrorists to Hell'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/response', methods=['post'])
def response():
    print("RESPONSE ...")

    # Loading model
    model = torch.load(model_path)
    model.eval()

    # Pre-processing inputs
    input_text = request.form.get('message')
    print(input_text)
    input_text = input_text.strip()
    if input_text == '':
        print("Oops! You input nothing!")
        emotions = ['NONE']
        movies = [['NONE']]
        songs = [['NONE']]
        return render_template('response.html', emotions=emotions, movies=movies, songs=songs)
    else:
        pro_sent = preprocessor(input_text)

        # Embedding and vectorize
        word2idx, _, embeddings = load_embeddings(model_conf)
        sample = vectorize(pro_sent, word2idx, max_length)

        # Processing to get model inputs
        samples = []
        lengths = []
        samples.append(sample)
        lengths.append(len(pro_sent))

        # tensor([17,  4, 37, 42, 21, 16, 13, 44, 29, 10, 15, 22, 21, 23, 18, 23, 25, 10,
        #         23,  9, 22, 18, 14, 15,  6, 33, 14, 30, 13, 22, 26, 17],
        #        device='cuda:0')
        # torch.ones([2, 4], dtype=torch.float64, device=cuda0)
        samples = np.asarray(samples)
        lengths = np.asarray(lengths)

        samples = torch.tensor(samples)
        lengths = torch.tensor(lengths)

        samples = samples.to(DEVICE)
        lengths = lengths.to(DEVICE)


        # Running model
        outputs, attentions = model(samples, lengths)

        # print(attentions)

        # tensor([ 2.1146, -1.7304,  2.0117, -1.3296, -3.1048, -5.9759, -2.7536, -2.7494,   print(outputs)
        #         -1.8445, -4.1412, -4.7449], device='cuda:0', grad_fn=<ThAddBackward>)
        # tensor([0.0521, 0.0631, 0.0632, 0.0632, 0.0632, 0.0632, 0.0631, 0.0632, 0.0632,   print(attentions)
        #         0.0632, 0.0631, 0.0632, 0.0632, 0.0632, 0.0632, 0.0632],
        #        device='cuda:0', grad_fn=<DivBackward1>)
        # gold: 1   1   0   0   0   0   1   0   0   0   1
        # [ 2.11464429 -1.7303592   2.01172185 -1.32956982 -3.10483193 -5.97593689  print(posts)
        #  -2.75357819 -2.74935317 -1.84453487 -4.14115143 -4.74489594]
        posts = outputs.data.cpu().numpy()
        predicted = np.clip(np.sign(posts), a_min=0, a_max=None)    # 1   1   0   0   0   0   1   0   0   0   1
        predicted = predicted.astype(np.int32)

        emotions = []
        ids = set()
        sum_item = 0
        for idx, item in enumerate(predicted):
            if item == 1:
                emotions.append(labels[idx])
                ids.add(recommend_id[idx])
            sum_item += item
        if sum_item == 0:
            emotions.append(labels[11])  # neutral
            ids.add(recommend_id[11])

        if len(emotions) > 6:
            print("Hey, there are more than 6 predicted emotions. Below are original emotions and emotions for displaying.")
            print(emotions)
            emotions = emotions[:6]
            print(emotions)
        else:
            print(emotions)

        print(ids)

        # movies and music matching
        all_movies = load_movies(movie_path)
        all_music = load_music(music_path)
        movies = []
        songs = []
        for i in ids:
            for m in all_movies[i]:
                movies.append(m)
            for s in all_music[i]:
                songs.append(s)

        print(songs)
        print(movies)

        if len(movies) > 3:
            random_int = set()
            while len(random_int) < 3:
                index = np.random.randint(low=0, high=len(movies))
                random_int.add(index)
            new_movies = []
            for i in random_int:
                new_movies.append(movies[i])
            movies = new_movies
            print(movies)
        if len(songs) > 6:
            random_int = set()
            while len(random_int) < 6:
                index = np.random.randint(low=0, high=len(songs))
                random_int.add(index)
            # indexs = np.random.randint(low=0, high=len(songs), size=6)
            new_songs = []
            for i in random_int:
                new_songs.append(songs[i])
            songs = new_songs
            print(songs)


        return render_template('response.html', emotions=emotions, movies=movies, songs=songs)


def main():
    print("Starting...")

    app.run(host='0.0.0.0', debug=False)


if __name__ == '__main__':
    main()
