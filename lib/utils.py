import os
import numpy
import errno
import pickle

# from tqdm import tqdm
from torch.utils.data import DataLoader

from lib import config
from lib.data_utils import WordDataset


def parse(dataset):
    data_file = config.E_C[dataset]

    with open(data_file, 'r') as fin:
        data = [l.strip().split('\t') for l in fin.readlines()][1:]

    tweets = [l[1] for l in data]
    labels = [[int(label) for label in l[2:]] for l in data]

    return tweets, labels


def write_cache_word_vectors(file, data):
    with open(file_cache_name(file), 'wb') as pickle_file:
        pickle.dump(data, pickle_file)


def file_cache_name(file):
    head, tail = os.path.split(file)
    filename, ext = os.path.splitext(tail)
    return os.path.join(head, filename + ".p")


def load_cache_word_vectors(file):
    with open(file_cache_name(file), 'rb') as f:
        return pickle.load(f)


def load_word_vectors(file, dim):
    """
    Read the word vectors from a text file
    Args:
        file: the filename
        dim: the dimensions of the word vectors

    Returns:
        word2idx (dict): dictionary of words to ids
        idx2word (dict): dictionary of ids to words
        embeddings (numpy.ndarray): the word embeddings matrix

    """
    # in order to avoid time consuming operation, detecting cache
    try:
        cache = load_cache_word_vectors(file)
        print("Loaded word embeddings from cache.")
        return cache
    except OSError:
        print("Didn't find embeddings cache file {}".format(file))

    # creating the necessary dictionaries and the word embeddings matrix
    if os.path.exists(file):
        print('Indexing file {} ...'.format(file))

        word2idx = {}  # dictionary of words to ids
        idx2word = {}  # dictionary of ids to words
        embeddings = []  # the word embeddings matrix

        # creating the 2D array,
        # which will be used for initializing the Embedding layer of a NN;
        # the first row (idx=0) is reserved as the zeros word embedding,
        # which will be used for zero padding (word with id = 0).
        embeddings.append(numpy.zeros(dim))
        word2idx["<padding>"] = 0
        idx2word[0] = "<padding>"

        # reading file line by line
        with open(file, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                values = line.split(" ")
                word = values[0]
                vector = numpy.asarray(values[1:], dtype='float32')
                index = i + 1

                idx2word[index] = word
                word2idx[word] = index
                embeddings.append(vector)

            # adding an unk token for OOV words
            if "<unk>" not in word2idx:
                idx2word[len(idx2word)] = "<unk>"
                word2idx["<unk>"] = len(word2idx)
                embeddings.append(
                    numpy.random.uniform(low=-0.05, high=0.05, size=dim))

            print(set([len(x) for x in embeddings]))
            print('Found %s word vectors.' % len(embeddings))

            embeddings = numpy.array(embeddings, dtype='float32')

        # write the data to a cache file
        write_cache_word_vectors(file, (word2idx, idx2word, embeddings))

        return word2idx, idx2word, embeddings

    else:
        print("{} not found!".format(file))
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT))

'''
def twitter_preprocess():
    def preprocess(name, dataset):
        desc = "Pre-processing dataset {}...".format(name)
        data = [preprocessor(x) for x in tqdm(dataset, desc=desc)]
        return data

    return preprocess
'''

'''
    datasets = {
        "train": (X_train, y_train),
        "dev": (X_dev, y_dev),
        "test": (X_test, y_test),
    }    
'''

def load_datasets(datasets, train_batch_size, eval_batch_size, token_type,
                  preprocessor=None, params=None, word2idx=None):
    name = params   # e.g. EmotionClassification_dev

    loaders = {}
    if token_type == "word":
        if word2idx is None:
            raise ValueError
        '''
        if preprocessor is None:
            preprocessor = twitter_preprocess()
        '''
        print("Building word-level datasets...")
        for type, Xy_data in datasets.items():  # train, test, dev
            _name = "{}_{}".format(name, type)
            dataset = WordDataset(Xy_data[0], Xy_data[1], word2idx, name=_name,     # Xy_data[0] = tweet; Xy_data[1] = labels
                                  preprocess=preprocessor)
            batch_size = train_batch_size if type == "train" else eval_batch_size
            loaders[type] = DataLoader(dataset, batch_size, shuffle=True,
                                       drop_last=True)
    else:
        raise ValueError("Invalid token type!")

    return loaders

def load_movies(data_path):
    # 1, The Sound of Music, 1965, https://www.imdb.com/title/tt0059742/, A woman leaves an Austrian convent to become a governess to the children of a Naval officer widower. 
    results = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    with open(data_path, 'r') as fin:
        for line in fin.readlines():
            items = line[:len(line)-1].split(',')
            label = int(items[0])
            movie = []
            name = items[1].strip()
            year = items[2].strip()
            web = items[3].strip()
            des = ','.join(items[4:]).strip()
            movie.append(name)
            movie.append(year)
            movie.append(web)
            movie.append(des)
            results[label].append(movie)
    return results

 
def load_music(data_path):
    # 6, You Are The Right One, Sports, You Are The Right One, https://music.163.com/#/song?id=553534022 
    results = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    with open(data_path, 'r') as fin:
        for line in fin.readlines():
            items = line[:len(line)-1].split(',')
            label = int(items[0])
            song = []
            name = items[1].strip()
            singer = items[2].strip()
            album = items[3].strip()
            web = items[4].strip()
            song.append(name)
            song.append(singer)
            song.append(album)
            song.append(web)
            results[label].append(song)
    return results
