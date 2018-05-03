from data_layer import FeaturizeData
from sklearn import linear_model
from sklearn.utils import class_weight
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(use_cuda)

class TextClassifier(nn.Module):
    def __init__(self, num_labels, num_features):
        super(TextClassifier, self).__init__()
        self.linear1 = nn.Linear(num_features, 600)
        self.linear2 = nn.Linear(600, 1200)
        self.linear3 = nn.Linear(1200, num_labels)


    def forward(self, x):
        x = self.linear1(x)
        x = F.tanh(x)
        x = self.linear2(x)
        x = F.tanh(x)
        x = self.linear3(x)
        return F.log_softmax(x, dim=1)


class TrainModel():
    def __init__(self, model, optimizer, dev_x, dev_y):
        self.model = model
        self.optimizer = optimizer
        self.dev_x = dev_x
        self.dev_y = dev_y

    def get_class_weights(self, train_y):
        train_y = train_y.cpu().data.numpy()
        weights = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)
        return torch.Tensor(weights).to(device)

    def train_model(self, train_x, train_y, epochs=1000):
        class_weights = self.get_class_weights(train_y)

        for i in range(epochs):
            if i % (epochs / 5) == 0: print("Epoch {0}".format(i))
            self.model.zero_grad()
            log_probs = self.model(train_x)
            # does this need to be a one hot vector - train_y
            loss = F.cross_entropy(log_probs, train_y, weight=class_weights)
            loss.backward()
            self.optimizer.step()

    def semi_predict(self, unlabeled_x):
        # maybe return the prob and decide which ones to add to the train
        with torch.no_grad():
            predicted_y = torch.argmax(self.model(unlabeled_x), dim=1)
            return predicted_y

    def evaluate_model(self):
        with torch.no_grad():
            output_y = self.model(self.dev_x)
            dev_loss = F.cross_entropy(output_y, self.dev_y).item()
            dev_predicted_y = torch.argmax(output_y, dim=1)
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

    network = TrainModel(text_classifier, optimizer, dev_tensor_x, dev_tensor_y)

    # train the model on labeled data
    network.train_model(train_tensor_x, train_tensor_y, epochs=2500)
    correct, dev_loss = network.evaluate_model()
    print("Correct preds: {0} Accuracy: {1}, dev_loss: {2}".format(100.0 * correct / dev_y.shape[0], correct, dev_loss))

    for unlabeled_x in featurize_data.get_unlabeled_feature_list(batch_size=487):
        print("unlabeled data loaded: {0}".format(unlabeled_x.shape))
        unlabeled_tensor_x = torch.Tensor(unlabeled_x).to(device)
        unlabeled_predicted_y = network.semi_predict(unlabeled_tensor_x)

        train_tensor_x = torch.cat((train_tensor_x, unlabeled_tensor_x))
        train_tensor_y = torch.cat((train_tensor_y, unlabeled_predicted_y))

        print("Train_X: {0}, Train_Y: {1}".format(train_tensor_x.size(), train_tensor_y.size()))

        # retrain model on new data tensors
        network.train_model(train_tensor_x, train_tensor_y, epochs=1000)
        correct, dev_loss = network.evaluate_model()
        print("Correct preds: {0} Accuracy: {1}, dev_loss: {2}".format(100.0 * correct / dev_y.shape[0], correct,
                                                                      dev_loss))


if __name__ == "__main__":
    main()
