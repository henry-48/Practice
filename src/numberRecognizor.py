import operator
import torch
from base64 import b64decode
from json import loads
import numpy as np
from matplotlib.pyplot import figure
import matplotlib as plt
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn


def read_data(json_file):
    json_object = loads(json_file)
    json_data = b64decode(json_object["data"])
    digit_vector = np.fromstring(json_data, dtype=np.ubyte)
    digit_vector = digit_vector.astype(np.float64)
    return json_object["label"], digit_vector


with open("digits.base64.json", "r") as f:
    digits = [read_data(each) for each in f.readlines()]

training_size = int(len(digits) * 0.25)
test = np.array(digits[:600])
validation = np.array(digits[600:training_size])
training = np.array(digits[training_size:])

X = list(training[:, 1])
print(type(X))
print(type(X[0]))

Y = np.array(training[:, 0]).astype("int")

X_val = list(validation[:, 1])
Y_val = np.array(validation[:, 0]).astype("int")

Test_X = list(test[:, 1])
Test_Y = np.array(test[:, 0]).astype("int")


# KNN
#
#
#
# clf = KNeighborsClassifier(n_neighbors=10)
# # clf.fit(X,Y)
# print("Test set accuracy: {:.2f}".format(clf.score(X_val, Y_val)))
#
# pre = clf.predict(Test_X)
#
#
# print(clf.score(Test_X, Test_Y))
#
#
#
# #Decision Tree
#
# tree = DecisionTreeClassifier(random_state=0) # overfit
# tree.fit(X,Y)
# print("Train set accuracy: {:.2f}".format(tree.score(X, Y)))
# print("Test set accuracy: {:.2f}".format(tree.score(X_val, Y_val)))
#
# tree = DecisionTreeClassifier(max_depth=5,random_state=0)
# tree.fit(X,Y)
# print("Train set accuracy: {:.2f}".format(tree.score(X, Y)))
# print("Test set accuracy: {:.2f}".format(tree.score(X_val, Y_val)))


# Naive Bayes

# nbM = MultinomialNB()
# nbM.fit(X,Y)
# print(nbM.score(X_val,Y_val))
#
# nbB = BernoulliNB()
# nbB.fit(X,Y)
# print("Bernoulli form: {:.2f}".format(nbB.score(X_val,Y_val)))


# KNN program
# def classifyKNN(testX, dataset, label, k):
#     datasetSize = dataset.shape[0]
#     diffMat = np.tile(testX, (datasetSize, 1)) - dataset
#     sqDiffMat = diffMat**2
#     sqDistance = sqDiffMat.sum(axis=1)
#     distance = sqDistance ** 0.5
#     sortDistance = distance.argsort()
#
#     classCount = {}
#     for i in range(k):
#         voteLabel = label[sortDistance[i]]
#         classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
#
#     sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
#
#     return sortedClassCount[0][0]
#
#
# predY = []
#
# a = np.array(X)
#
# for testX in Test_X:
#     predY.append(classifyKNN(testX, a, Y, 6))
#
# score = 0
# for i in range(len(predY)):
#     if predY[i] == Test_Y[i]:
#         score += 1
#
# print(score)
# print("KNN score is {:.2f}".format(score / len(Test_Y)))
#


# SVM
# clf = SVC(C = 5)
# clf.fit(X,Y)
#
# print("SVM sklearn RBF kernal: {:.2f}".format(clf.score(X_val,Y_val)))

# clf = SVC(C=5, kernel= 'sigmoid')
# clf.fit(X,Y)
# print("SVM sklearn Sigmoid kernal: {:.2f}".format(clf.score(X_val,Y_val)))

# clf = SVC(C=5, kernel= 'linear')
# clf.fit(X,Y)
# print("SVM sklearn Linear kernal: {:.2f}".format(clf.score(X_val,Y_val)))


# Neural necwork

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class NUMDataset(Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
        if mode == 'train':
            self.X_data = torch.tensor(np.array(X))
            print(self.X_data.shape)
            self.Y_data = torch.tensor(np.array(Y))
            print(self.Y_data.shape)
        elif mode == 'val':
            self.X_data = torch.tensor(np.array(X_val))
            self.Y_data = torch.tensor(np.array(Y_val))
        else:
            self.X_data = torch.tensor(np.array(Test_X))
            self.Y_data = torch.tensor(np.array(Test_Y))

    def __getitem__(self, index):
        if self.mode in ['train', 'val']:
            return self.X_data[index], self.Y_data[index]
        else:
            return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


def prep_dataLoader(mode, batch_size):
    dataset = NUMDataset(mode)
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=True
    )
    return dataloader


BATCH_SIZE = 1
train_len = NUMDataset('train').__len__()
train_loader = prep_dataLoader(mode='train', batch_size=BATCH_SIZE)
val_len = NUMDataset('val').__len__()
val_loader = prep_dataLoader(mode='val', batch_size=BATCH_SIZE)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.layer1 = nn.Linear(784, 2048)
        self.layer2 = nn.Linear(2048, 512)
        self.layer3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 10)

        self.act_fn = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.act_fn(x)



        x = self.out(x)

        return x


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(1)
device = get_device()
print(f'DEVICE: {device}')
num_epoch = 20
learning_rate = 0.001

model_path = './model.ckpt'

model = Classifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_acc = 0.0

for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        inputs = inputs.to(torch.float32)
        output = model(inputs)

        labels = labels.to(torch.Tensor().long())
        batch_loss = criterion(output, labels)
        _, train_pred = torch.max(output, 1)
        batch_loss.backward()
        optimizer.step()


        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()


        train_loss = batch_loss.item()


    if len(X_val) > 0:
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                inputs = inputs.to(torch.float32)
                output = model(inputs)
                labels = labels.to(torch.Tensor().long())
                batch_loss = criterion(output, labels)
                _, val_pred = torch.max(output, 1)

                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                val_loss = batch_loss.item()
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc / train_len, train_loss / train_len, val_acc / val_len,
                val_loss / val_len
            ))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc / len(X_val)))



    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc / len(X), train_loss / len(train_loader)
        ))

if len(X_val) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')

