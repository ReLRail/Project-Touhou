import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from numpy import asarray
import glob

from torch.utils.data import TensorDataset, DataLoader

file_list = list(glob.glob('data/exp2/*.png'))
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

if 0:

    dict = {
        'z': 0,
        'shift_up_z': 1,
        'shift_left_z': 2,
        'shift_down_z': 3,
        'shift_right_z': 4
        }

    X = asarray(tuple((asarray(Image.open(x).convert('L')) for x in file_list )))/255
    y = asarray(tuple((dict[x[x.find('_')+1:x.find('.')]] for x in file_list)))

    X = asarray([X.astype(float)])
    X = np.transpose(X, (1,0,2,3))

    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    del X
    del y

    #print(X_train, X_test, y_train, y_test)
    print(len(X_train), len(X_test), len(y_train), len(y_test))
    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    torch.save(X_train, 'tensor/X_train.pt')
    torch.save(X_test, 'tensor/X_test.pt')
    torch.save(y_train, 'tensor/y_train.pt')
    torch.save(y_test, 'tensor/y_test.pt')

    print(X_train.shape)
    print(y_train.shape)

X_train = torch.load('tensor/X_train.pt').cuda()
X_test = torch.load('tensor/X_test.pt').cuda()
y_train = torch.load('tensor/y_train.pt').cuda()
y_test = torch.load('tensor/y_test.pt').cuda()

print(torch.cuda.memory_allocated())
print(torch.cuda.memory_summary())


my_dataset = TensorDataset(X_train,y_train) # create your datset
my_dataset_val = TensorDataset(X_train,y_train) # create your datset
trainloader = DataLoader(my_dataset, batch_size=4) # create your dataloader
test_loader = DataLoader(my_dataset_val, batch_size=256) # create your dataloader

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
net = models.resnet18(pretrained=True)
net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7),stride=(2, 2),padding=(3, 3), bias=False)
net = net.cuda()
print(net)
print(summary(net,(1, 480, 640)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
loss_log = []
loss_val = []
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = []
    net.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss.append(loss.item())
    running_loss_val = []
    with torch.no_grad():
        net.eval()
        for i, data in enumerate(test_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # print statistics
            running_loss_val.append(loss.item())

    loss_log.append(sum(running_loss)/len(running_loss))
    loss_val.append(sum(running_loss_val)/len(running_loss_val))
    print('epoch:',epoch, 'loss', sum(running_loss)/len(running_loss),'val_loss',sum(running_loss_val)/len(running_loss_val))
plt.plot(loss_log)
plt.plot(loss_val)
plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
print('Finished Training')

PATH = './Touhou.pth'
torch.save(net.state_dict(), PATH)