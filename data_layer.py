import os
import gensim
import numpy as np
import io
import re
from sklearn.feature_extraction.text import TfidfVectorizer


class Sentences():
    def __init__(self, dirnames: list):
        self.dirnames = dirnames

    def __iter__(self):
        # i = 0
        for dirname in self.dirnames:
            for fname in os.listdir(dirname):
                for line in io.open(os.path.join(dirname, fname), 'r', encoding='utf-8'):
                    # TODO: remove common words? lowercase? preprocess...
                    # print("Done: {0}, fname {1} ".format(i, fname))
                    line = line.split()
                    # # print(line)
                    line = [''.join(re.findall("[a-zA-Z0-9]+", word)) for word in line]
                    # print(line)
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
    def __init__(self, model_filename, train_filename, dev_filename, labeled_data_filename, unlabeled_data_filename):
        # self.model = gensim.models.KeyedVectors.load_word2vec_format(model_filename, binary=True)
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.labeled_data_filename = labeled_data_filename
        self.unlabeled_data_filename = unlabeled_data_filename


    def get_trainset(self):
        pass



# temp_file = io.open('./data/tempfile', 'w')
# sentences = Sentences(['./data/A3/labeled/'])
# for i, s in enumerate(sentences):
#     temp_file.write(s)
#     # print(s)
#     if i == 50:
#         break

tfidf = TrainTFIDF('./data/A3/labeled', './data/A3/unlabeled')
weights_dict = tfidf.get_tfidf()
