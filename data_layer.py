import os
import gensim
import numpy as np
import io
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl
import csv


class Sentences():
    def __init__(self, dirnames: list):
        self.dirnames = dirnames

    def __iter__(self):
        # i = 0
        for dirname in self.dirnames:
            for fname in os.listdir(dirname):
                for line in io.open(os.path.join(dirname, fname), 'r', encoding='utf-8'):
                    # TODO: remove common words? lowercase? preprocess...
                    line = line.split()
                    line = [''.join(re.findall("[a-zA-Z0-9]+", word)) for word in line]
                    yield line


class SentenceFileItr():
    def __init__(self, file_list: list):
        self.file_list = file_list

    def __iter__(self):
        for fname in self.file_list:
            for line in io.open(fname, 'r', encoding='utf-8'):
                line = line.split()
                line = [''.join(re.findall("[a-zA-Z0-9]+", word)) for word in line]
                yield line


class TrainTFIDF():
    def __init__(self, labeled_data_dirname, unlabeled_data_dirname):
        self.labeled_data_dirname = labeled_data_dirname
        self.unlabeled_data_dirname = unlabeled_data_dirname
        self.sentences = Sentences([self.labeled_data_dirname, self.unlabeled_data_dirname])

    def collect_sentences(self):
        sentences_list = []
        for i, line in enumerate(self.sentences):
            print("Done:", i)
            sentences_list.append(' '.join(line))
        return sentences_list

    def get_tfidf(self):
        sentence_list = self.collect_sentences()
        vectorizer = TfidfVectorizer(min_df=1)
        vectorizer.fit_transform(sentence_list)
        idf = vectorizer.idf_
        idf = idf / np.sum(idf)
        weights_dict = dict(zip(vectorizer.get_feature_names(), idf))
        return weights_dict


class FeaturizeData():
    def __init__(self, model_filename, tfidf_model_filename, train_filename, dev_filename,
                 labeled_data_dirname, unlabeled_data_dirname):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_filename, binary=True)
        self.tfidf_model = pkl.load(io.open(tfidf_model_filename, 'rb'))
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.labeled_data_dirname = labeled_data_dirname
        self.unlabeled_data_dirname = unlabeled_data_dirname
        print("models loaded")


    def get_trainset(self):
        train_data_files = []
        train_labels = []
        with io.open(self.train_filename, 'r', encoding='utf-8') as train_file:
            reader = csv.reader(train_file, delimiter='\t')
            for row in reader:
                train_data_files += [os.path.join(self.labeled_data_dirname, os.path.basename(row[0]))]
                train_labels += [row[1]]

        train_files_itr = SentenceFileItr(train_data_files)
        train_x = np.zeros((len(train_labels), 300)) # change this to num features
        for i, sentence in enumerate(train_files_itr):
            train_x[i] = self.get_feature_vector(sentence)
            # print(train_x[i])
            break
        exit(1)

        return train_x, train_labels


    def get_devset(self):
        dev_data_files = []
        dev_labels = []
        with io.open(self.dev_filename, 'r', encoding='utf-8') as dev_file:
            reader = csv.reader(dev_file, delimiter='\t')
            for row in reader:
                dev_data_files += [os.path.join(self.labeled_data_dirname, os.path.basename(row[0]))]
                dev_labels += [row[1]]
        dev_files_itr = SentenceFileItr(dev_data_files)
        dev_x = np.zeros((len(dev_labels), 300))
        for i, sentence in enumerate(dev_files_itr):
            dev_x[i] = self.get_feature_vector(sentence)
        return dev_x, dev_labels


    def get_feature_vector(self, word_list):
        print(word_list)
        word_feature = np.zeros((300)) # change this to num features
        i = 0
        for word in word_list:
            try:
                print("word {0} tfifd {1}".format(word, self.tfidf_model[word]))
                word_feature += self.tfidf_model[word] * self.model[word]
            except:
                i+=1
                print("here with word", word)
                continue

        print("words found: {0}, words not found {1}".format(len(word_list) - i, i))
        return word_feature


# tfidf = TrainTFIDF('./data/A3/labeled', './data/A3/unlabeled')
# weights_dict = tfidf.get_tfidf()
# weights_file = io.open("models/tfidf_weights.pkl", 'wb')
# pkl.dump(weights_dict, weights_file)


fd = FeaturizeData('models/google_word2vec.bin', 'models/tfidf_weights.pkl', 'data/A3/train.tsv',
                   'data/A3/dev.tsv', 'data/A3/labeled', 'data/A3/unlabeled')
fd.get_trainset()