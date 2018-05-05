from data_layer import FeaturizeData
from sklearn import linear_model
from sklearn.utils import class_weight
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import io

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(use_cuda)
BATCH_SIZE = 487

class TextClassifier(nn.Module):
    def __init__(self, num_labels, num_features):
        super(TextClassifier, self).__init__()
        self.linear1 = nn.Linear(num_features, 600)
        self.linear2 = nn.Linear(600, num_labels)

    def forward(self, x):
        x = self.linear1(x)
        x = F.tanh(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)


class TrainModel():
    def __init__(self, model, optimizer, dev_x, dev_y, label_list):
        self.model = model
        self.optimizer = optimizer
        self.dev_x = dev_x
        self.dev_y = dev_y
        self.label_list = label_list

    def get_class_weights(self, train_y):
        train_y = train_y.cpu().data.numpy()
        weights = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)
        return torch.Tensor(weights).to(device)

    def train_model(self, train_x, train_y, epochs=1000):
        class_weights = self.get_class_weights(train_y)

        for i in range(epochs):
            self.model.zero_grad()
            log_probs = self.model(train_x)
            loss = F.nll_loss(log_probs, train_y, weight=class_weights)
            if (i + 1) % (epochs / 2) == 0: print("Epoch {0}, loss {1}".format(i+1, loss.item()))
            loss.backward()
            self.optimizer.step()

    def semi_predict(self, unlabeled_x):
        # maybe return the prob and decide which ones to add to the train
        with torch.no_grad():
            predicted_y = torch.argmax(self.model(unlabeled_x), dim=1)
            return predicted_y

    def evaluate_model(self, print_stats=False):
        with torch.no_grad():
            output_y = self.model(self.dev_x)
            dev_loss = F.nll_loss(output_y, self.dev_y).item()
            dev_predicted_y = torch.argmax(output_y, dim=1)
            if print_stats:
                print(classification_report(self.dev_y.cpu().data.numpy(), dev_predicted_y.cpu().data.numpy(), target_names=self.label_list))
            correct_labels = dev_predicted_y.eq(self.dev_y).sum().item()
            return correct_labels, dev_loss


def main():
    featurize_data = FeaturizeData('models/google_word2vec.bin', 'models/tfidf_weights.pkl', 'data/A3/train.tsv',
                                        'data/A3/dev.tsv', 'data/A3/labeled', 'data/A3/unlabeled')

    train_x, train_y = featurize_data.get_trainset()
    train_tensor_x = torch.Tensor(train_x).to(device)
    train_tensor_y = torch.LongTensor(train_y).to(device)

    dev_x, dev_y = featurize_data.get_devset()
    dev_tensor_x = torch.Tensor(dev_x).to(device)
    dev_tensor_y = torch.LongTensor(dev_y).to(device)

    text_classifier = TextClassifier(len(featurize_data.label_to_idx.values()), train_x.shape[1]).to(device)
    optimizer = optim.SGD(text_classifier.parameters(), lr=0.1)

    label_list = [featurize_data.get_idx_to_label(i) for i in range(np.max(dev_tensor_y.cpu().data.numpy()))]
    network = TrainModel(text_classifier, optimizer, dev_tensor_x, dev_tensor_y, label_list)

    # train the model on labeled data
    network.train_model(train_tensor_x, train_tensor_y, epochs=1000)
    correct, dev_loss = network.evaluate_model(print_stats=True)
    accuracy = 100.0 * correct / dev_y.shape[0]
    print("Accuracy: {0} Correct Preds: {1}, dev_loss: {2}".format(accuracy, correct, dev_loss))

    i = 0
    acc_stats = np.zeros((5))
    percentages = [0, 0.10, 0.20, 0.50, 0.98]
    num_files = [i * 43343 for i in percentages]
    print(num_files)
    acc_stats[i] = accuracy
    i+=1
    for (num_itr, (unlabeled_x, _)) in enumerate(featurize_data.get_unlabeled_feature_list(batch_size=BATCH_SIZE)):
        num_points = (num_itr + 1) * BATCH_SIZE
        # print("unlabeled data loaded: {0}".format(unlabeled_x.shape))
        unlabeled_tensor_x = torch.Tensor(unlabeled_x).to(device)
        unlabeled_predicted_y = network.semi_predict(unlabeled_tensor_x)

        train_tensor_x = torch.cat((train_tensor_x, unlabeled_tensor_x))
        train_tensor_y = torch.cat((train_tensor_y, unlabeled_predicted_y))

        # print("Train_X: {0}, Train_Y: {1}".format(train_tensor_x.size(), train_tensor_y.size()))

        # retrain model on new data tensors
        network.train_model(train_tensor_x, train_tensor_y, epochs=1000)
        correct, dev_loss = network.evaluate_model()
        accuracy = 100.0 * correct / dev_y.shape[0]
        print("Num points processed {0}".format(num_points))
        if num_points >= num_files[i]:
            acc_stats[i] = accuracy
            print("Accuracy stats", acc_stats)
            i += 1
        print("Accuracy: {0} Correct Preds: {1}, dev_loss: {2}".format(accuracy, correct,
                                                                      dev_loss))
    plt.plot(np.array(percentages), acc_stats)
    plt.xlabel('Percentage')
    plt.ylabel('Dev Set Classification Accuracy')
    plt.savefig('./accuracy_2.png')

    classification_labels = []
    for (k, (unlabeled_x, unlabeled_files)) in enumerate(featurize_data.get_unlabeled_feature_list(batch_size=BATCH_SIZE)):
        unlabeled_tensor_x = torch.Tensor(unlabeled_x).to(device)
        unlabeled_predicted_y = network.semi_predict(unlabeled_tensor_x)

        predictions = unlabeled_predicted_y.cpu().data.numpy()
        for i, file in enumerate(unlabeled_files):
            classification_labels += [[file, featurize_data.get_idx_to_label(predictions[i])]]
        if (k + 1) % 10 == 0:
            print("Done with {0}".format((k+1) * BATCH_SIZE))
            save_file(classification_labels)
            classification_labels = []


def save_file(labels):
    df = pd.DataFrame(np.array(labels))
    with io.open('final_table_2.tsv', 'a', encoding='utf-8') as file:
        df.to_csv(file, sep='\t', header=None, index=False)

if __name__ == "__main__":
    main()
