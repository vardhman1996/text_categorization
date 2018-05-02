from data_layer import FeaturizeData
from sklearn import linear_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TextClassifier(nn.Module):
    def __init__(self, num_labels, num_features):
        super(TextClassifier, self).__init__()
        self.linear = nn.Linear(num_features, num_labels)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=1)


def train_model():
    featurize_data = FeaturizeData('models/google_word2vec.bin', 'models/tfidf_weights.pkl', 'data/A3/train.tsv',
                                        'data/A3/dev.tsv', 'data/A3/labeled', 'data/A3/unlabeled')

    train_x, train_y = featurize_data.get_trainset()
    train_tensor_x = torch.Tensor(train_x)
    train_tensor_y = torch.LongTensor(train_y)

    dev_x, dev_y = featurize_data.get_devset()
    dev_tensor_x = torch.Tensor(dev_x)
    dev_tensor_y = torch.LongTensor(dev_y)


    # logreg = linear_model.LogisticRegression(C=1e2, solver='lbfgs', max_iter=1000, multi_class='multinomial')
    # logreg.fit(train_x, train_y)

    text_classifier = TextClassifier(len(featurize_data.label_to_idx.values()), train_x.shape[1])
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(text_classifier.parameters(), lr=0.1)

    for epoch in range(1000):
        text_classifier.zero_grad()
        log_probs = text_classifier(train_tensor_x)

        loss = loss_function(log_probs, train_tensor_y)
        loss.backward()
        optimizer.step()

    print("trained model")


def main():
    train_model()


if __name__ == "__main__":
    main()
