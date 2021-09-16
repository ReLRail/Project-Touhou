import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from numpy import asarray
import glob
file_list = list(glob.glob('data/exp1/*.png'))
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

dict = {
    'z': 0,
    'shift_up_z': 1,
    'shift_left_z': 2,
    'shift_down_z': 3,
    'shift_right_z': 4
    }

X = asarray(tuple((asarray(Image.open(x).convert('RGB')) for x in file_list )))/255
y = asarray(tuple((dict[x[x.find('_')+1:x.find('.')]] for x in file_list)))

print(len(file_list), file_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train, X_test, y_train, y_test)
print(len(X_train), len(X_test), len(y_train), len(y_test))
X_train = torch.tensor(np.transpose(X_train, (0,3,1,2))).float().cuda()
X_test = torch.tensor(np.transpose(X_test, (0,3,1,2))).float().cuda()
y_train = torch.tensor(y_train, dtype=torch.long).cuda()
y_test = torch.tensor(y_test, dtype=torch.long).cuda()
torch.save(X_train, 'tensor/X_train.pt')
torch.save(X_test, 'tensor/X_test.pt')
torch.save(y_train, 'tensor/y_train.pt')
torch.save(y_test, 'tensor/y_test.pt')
print(X_train.shape)
print(y_train)

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

net = models.resnet18(pretrained=True).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


for epoch in range(50):  # loop over the dataset multiple times


    # get the inputs; data is a list of [inputs, labels]
    inputs = X_train
    labels = y_train
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    # print statistics
    print(epoch,loss.item())

print('Finished Training')

PATH = './Touhou.pth'
torch.save(net.state_dict(), PATH)