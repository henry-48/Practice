import struct
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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

    image = torch.tensor(np.empty((num_image, num_rows, num_cols)))

    for i in range(num_image):
        if (i + 1) % 1000 == 0:
            print('已解析%d' % (i + 1) + '张')
            print(offset)
        image[i] = torch.tensor(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
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
    labels = torch.tensor(np.empty(num_image))
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

Test_X = load_test_images()
Test_Y = load_test_labels()

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

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )


    def forward(self,x):
        x = self.cnn_layers(x)

        x = x.flatten(1)

        x = self.fc_layer(x)

        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Classifier().to(device)
model.device =device

criterion = nn.CrossEntropyLoss()

optimizer =torch.optim.Adam(model.parameters(),lr = 0.00001, weight_decay=1e-4)

n_epoch = 20

for epoch in range(n_epoch):
    model.train()
    train_loss = []
    train_acc = []

    for batch in tqdm(train_loader):
        imgs, labels =batch
        imgs = torch.unsqueeze(imgs,0)
        imgs = imgs.to(torch.float32)
        imgs,labels = imgs.to(device),labels.to(device)

        outputs = model(imgs)

        loss = criterion(outputs,labels.long())

        optimizer.zero_grad()

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm(model.parameters(),max_norm=10)
        optimizer.step()

        acc = (outputs.argmax(dim = -1) == labels).float().mean()

        train_loss.append(loss.item())
        train_acc.append(acc)

    train_loss =sum(train_loss)/len(train_loss)
    train_acc = sum(train_acc)/len(train_acc)

    print(f"[ Train | {epoch + 1:03d}/{n_epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    model.eval()

    valid_loss = []
    valid_acc = []

    for batch in tqdm(val_loader):
        imgs, labels = batch
        imgs = torch.unsqueeze(imgs, 0)
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(imgs)

        loss =criterion(outputs,labels)

        acc = (outputs.argmax(dim=-1) == labels.to(device)).float().mean()

        valid_loss.append(loss.item())
        valid_acc.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_acc) / len(valid_acc)

        print(f"[ Valid | {epoch + 1:03d}/{n_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


