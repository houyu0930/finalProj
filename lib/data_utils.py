import os
import pickle
import numpy

from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset

from lib.config import BASE_PATH
from lib.tokenizer import preprocessor

def vectorize(sequence, word2idx, max_length, unk_policy="random",
              spell_corrector=None):
    """
    Covert array of tokens, to array of ids, with a fixed length
    and zero padding at the end

    Args:
        sequence: a list of elements
        word2idx: dictionary of word to ids
        unk_policy (): how to handle OOV words
        spell_corrector (): if unk_policy = 'correct' then pass a callable
            which will try to apply spell correction to the OOV token

    Returns: list of ids with zero padding at the end

    """
    words = numpy.zeros(max_length).astype(int)

    # trimming tokens after max length
    sequence = sequence[:max_length]

    for i, token in enumerate(sequence):
        if token in word2idx:
            words[i] = word2idx[token]
        else:
            if unk_policy == "random":
                words[i] = word2idx["<unk>"]
            elif unk_policy == "zero":
                words[i] = 0
            elif unk_policy == "correct":
                corrected = spell_corrector(token)
                if corrected in word2idx:
                    words[i] = word2idx[corrected]
                else:
                    words[i] = word2idx["<unk>"]

    return words

class BaseDataset(Dataset):
    """
    This is a Base class which extends pytorch's Dataset, in order to avoid
    boilerplate code and equip our datasets with functionality such as caching.

    """
    def __init__(self, X, y,
                 max_length=0,
                 name=None,
                 label_transformer=None,
                 verbose=True,
                 preprocess=None):
        self.data = X
        self.labels = y
        self.name = name    # e.g. EmotionClassification_dev
        self.label_transformer = label_transformer

        if preprocess is not None:
            self.preprocess = preprocess

        self.data = self.load_preprocessed_data()

        self.set_max_length(max_length)

        if verbose:
            self.dataset_statistics()

    def set_max_length(self, max_length):
        # if max_length == 0, then set max_length
        # to the maximum sentence length in the dataset
        if max_length == 0:
            self.max_length = max([len(x) for x in self.data])
        else:
            self.max_length = max_length
        print('max length (words)')
        print(self.max_length)

    def dataset_statistics(self):
        raise NotImplementedError

    def preprocess(self, name, X):
        raise NotImplementedError

    @staticmethod
    def _check_cache():
        cache_dir = os.path.join(BASE_PATH, "_cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_filename(self):
        return os.path.join(BASE_PATH, "_cache",
                            "preprocessed_{}.p".format(self.name))

    def _write_cache(self, data):
        self._check_cache()

        cache_file = self._get_cache_filename()

        with open(cache_file, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

    def load_preprocessed_data(self):
        # NOT using cache
        if self.name is None:
            print("cache deactivated!")
            return self.preprocess(self.name, self.data)

        # using cache
        cache_file = self._get_cache_filename()

        if os.path.exists(cache_file):
            print("Loading {} from cache!".format(self.name))
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("No cache file for {} ...".format(self.name))
            data = self.preprocess(self.name, self.data)
            self._write_cache(data)
            return data


class WordDataset(BaseDataset):

    def __init__(self, X, y, word2idx,
                 max_length=0,
                 name=None,
                 label_transformer=None,
                 verbose=True,
                 preprocess=None):
        self.word2idx = word2idx

        BaseDataset.__init__(self, X, y, max_length, name, label_transformer,
                             verbose, preprocess)

    def dataset_statistics(self):
        words = Counter()
        for x in self.data: # data = X
            words.update(x)
        unks = {w: counts for w, counts in words.items() if w not in self.word2idx}
        # unks = sorted(unks.items(), key=lambda x: x[1], reverse=True)
        # print(unks)

        total_words = sum(words.values())
        total_unks = sum(unks.values())

        print("Total words: {}, Total unks:{} ({:.2f}%)".format(
            total_words, total_unks, total_unks * 100 / total_words))

        print("Unique words: {}, Unique unks:{} ({:.2f}%)".format(
            len(words), len(unks), len(unks) * 100 / len(words)))

        # label statistics
        print("Labels statistics:") # self.labels = y
        label_counts = {"anger":0, "anticipation":0, "disgust":0, "fear":0,
                  "joy":0, "love":0, "optimism":0, "pessimism":0, "sadness":0,
                  "surprise":0, "trust":0, "neutral":0}
        for label in self.labels:
            # print(label)
            label_counts["anger"] += label[0]
            label_counts["anticipation"] += label[1]
            label_counts["disgust"] += label[2]
            label_counts["fear"] += label[3]
            label_counts["joy"] += label[4]
            label_counts["love"] += label[5]
            label_counts["optimism"] += label[6]
            label_counts["pessimism"] += label[7]
            label_counts["sadness"] += label[8]
            label_counts["surprise"] += label[9]
            label_counts["trust"] += label[10]
            if (label[0] + label[1] + label[2] + label[3] + label[4] + label[5] + label[6]
                    + label[7] + label[8] + label[9] + label[10]) == 0:
                label_counts["neutral"] += 1
            # break

        label_percent = {}
        label_list = ["anger", "anticipation", "disgust", "fear", "joy",
                      "love", "optimism", "pessimism", "sadness", "surprise",
                      "trust", "neutral"]
        for each in label_list:
            label_percent[each] = label_counts[each] * 100 / len(self.labels)

        print(label_counts)
        print(label_percent)

    def preprocess(self, name, dataset):
        desc = "Pre-processing dataset {}...".format(name)
        data = [preprocessor(x) for x in tqdm(dataset, desc=desc)]
        return data

    # in order to let the DataLoader(PyTorch) know the size
    # of the datasets and to perform batching, shuffling and so on...
    def __len__(self):
        return len(self.data)

    # returning the properly processed data-item from the dataset with a given index
    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index(int)

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training sample
                * label (string): the class label
                * length (int): the length (tokens) of the sentence
                * index (int): the index of the dataitem in the dataset.
                               It is useful for getting the raw input for visualizations.
        """
        sample, label = self.data[index], self.labels[index]

        # transforming the sample and the label,
        # in order to feed them to the model
        sample = vectorize(sample, self.word2idx, self.max_length)

        '''
        if self.label_transformer is not None:
            label = self.label_transformer.transform(label)       
        '''
        if isinstance(label, (list, tuple)):
            label = numpy.array(label)


        return sample, label, len(self.data[index]), index
