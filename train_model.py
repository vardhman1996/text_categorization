from data_layer import FeaturizeData
from sklearn import linear_model
from sklearn.utils import class_weight
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


def train_model():
    featurize_data = FeaturizeData('models/google_word2vec.bin', 'models/tfidf_weights.pkl', 'data/A3/train.tsv',
                                        'data/A3/dev.tsv', 'data/A3/labeled', 'data/A3/unlabeled')

    train_x, train_y = featurize_data.get_trainset()
    weights = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)
    weights = torch.Tensor(weights / np.sum(weights))
    train_tensor_x = torch.Tensor(train_x)
    train_tensor_y = torch.LongTensor(train_y)

    dev_x, dev_y = featurize_data.get_devset()
    dev_tensor_x = torch.Tensor(dev_x)
    dev_tensor_y = torch.LongTensor(dev_y)

    text_classifier = TextClassifier(len(featurize_data.label_to_idx.values()), train_x.shape[1])
    loss_function = nn.NLLLoss(weight=weights)
    optimizer = optim.SGD(text_classifier.parameters(), lr=0.2)

    for epoch in range(2500):
        if epoch % 200 == 0: print("Epoch: ", epoch)
        text_classifier.zero_grad()
        log_probs = text_classifier(train_tensor_x)

        loss = loss_function(log_probs, train_tensor_y)
        loss.backward()
        optimizer.step()


    dev_predicted_y = torch.argmax(text_classifier(dev_tensor_x), dim=1)
    wrong_preds = torch.nonzero(dev_predicted_y - dev_tensor_y)

    print("wrongs: ", wrong_preds.size())
    print("totals: ", dev_tensor_y.size())


def main():
    train_model()


if __name__ == "__main__":
    main()
