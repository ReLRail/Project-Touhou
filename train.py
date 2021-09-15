import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import glob
file_list = list(glob.glob('data/exp2/*.png'))

from PIL import Image

dict = {
    'z': 0,
    'shift_up_z': 1,
    'shift_left_z': 2,
    'shift_down_z': 3,
    'shift_right_z': 4
    }

file_list = tuple(((dict[x[x.find('_')+1:x.find('.')]],Image.open(x).convert('RGB'))for x in file_list))

print(file_list)

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))



import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

net = models.resnext101_32x8d(pretrained=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
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
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)