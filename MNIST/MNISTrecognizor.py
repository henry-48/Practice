import struct
import numpy as np
import torch
from torch import Tensor

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

train_image = 'train-images.idx3-ubyte'
train_lable = 'train-labels.idx1-ubyte'
test_image = 't10k-images.idx3-ubyte'
test_label = 't10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, "rb").read()
    offset = 0
    fmt_header = ">iiii"
    magic_number, num_image, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d,图片数量:%d,图片大小：%d*%d' % (magic_number, num_image, num_rows, num_cols))

    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    print(offset)
    fmt_image = '>' + str(image_size) + 'B'
    print(fmt_image, offset, struct.calcsize(fmt_image))

    image = np.empty((num_image, num_rows, num_cols))

    for i in range(num_image):
        if (i + 1) % 1000 == 0:
            print('已解析%d' % (i + 1) + '张')
            print(offset)
        image[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)

    return image


def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>ii'
    magic_number, num_image = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d,图片数量:%d' % (magic_number, num_image))

    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_image)
    for i in range(num_image):
        if (i + 1) % 10000 == 0:
            print("已解析%d" % (i + 1) + "张")
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)

    return labels


def load_train_images(idx_ubyte_file=train_image):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_lable):
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_image):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_label):
    return decode_idx1_ubyte(idx_ubyte_file)



X = load_train_images()
Y = load_train_labels()
X = X.reshape(60000,784)

Test_X = load_test_images()
Test_X = Test_X.reshape(10000,784)
Test_Y = load_test_labels()


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class NUMDataset(Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
        if mode == 'train':
            self.X_data = torch.tensor(np.array(X))
            self.Y_data = torch.tensor(np.array(Y))

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


BATCH_SIZE = 50
train_len = NUMDataset('train').__len__()
train_loader = prep_dataLoader(mode='train', batch_size=BATCH_SIZE)
val_len = NUMDataset('val').__len__()
val_loader = prep_dataLoader(mode='val', batch_size=BATCH_SIZE)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.layer1 = nn.Linear(784, 2000)
        self.layer2 = nn.Linear(2000, 500)
        self.layer3 = nn.Linear(500, 100)
        self.out = nn.Linear(100, 10)

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

        train_loss += batch_loss.item()

    if len(Test_X) > 0:
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
                val_loss += batch_loss.item()
            print("Correct Number : {:f}, Total Number: {:f} ".format(val_acc,val_len))
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc / train_len, train_loss / train_len, val_acc / val_len,
                val_loss / val_len
            ))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc / val_len))



    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc / train_len, train_loss / train_len
        ))

if val_len == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')