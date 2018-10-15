import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time

from tqdm import tqdm

from model.multi_scale_ori import *
# from multi_scale_nores import *
# from multi_scale_one3x3 import *
# from multi_scale_one5x5 import *
# from multi_scale_one7x7 import *


batch_size = 1024
num_epochs = 350


# load data
data = sio.loadmat('data/changingSpeed_train.mat')
train_data = data['train_data_split']
train_label = data['train_label_split']

num_train_instances = len(train_data)

train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
train_label = torch.from_numpy(train_label).type(torch.LongTensor)
train_data = train_data.view(num_train_instances, 1, -1)
train_label = train_label.view(num_train_instances, 1)

train_dataset = TensorDataset(train_data, train_label)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



data = sio.loadmat('data/changingSpeed_test.mat')
test_data = data['test_data_split']
test_label = data['test_label_split']

num_test_instances = len(test_data)

test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
test_label = torch.from_numpy(test_label).type(torch.LongTensor)
test_data = test_data.view(num_test_instances, 1, -1)
test_label = test_label.view(num_test_instances, 1)

test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


msresnet = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=6)
msresnet = msresnet.cuda()

criterion = nn.CrossEntropyLoss(size_average=False).cuda()

optimizer = torch.optim.Adam(msresnet.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250, 300], gamma=0.1)
train_loss = np.zeros([num_epochs, 1])
test_loss = np.zeros([num_epochs, 1])
train_acc = np.zeros([num_epochs, 1])
test_acc = np.zeros([num_epochs, 1])

for epoch in range(num_epochs):
    print('Epoch:', epoch)
    msresnet.train()
    scheduler.step()
    # for i, (samples, labels) in enumerate(train_data_loader):
    loss_x = 0
    for (samples, labels) in tqdm(train_data_loader):
        samplesV = Variable(samples.cuda())
        labels = labels.squeeze()
        labelsV = Variable(labels.cuda())

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        predict_label = msresnet(samplesV)

        loss = criterion(predict_label[0], labelsV)

        loss_x += loss.item()

        loss.backward()
        optimizer.step()

    train_loss[epoch] = loss_x / num_train_instances

    msresnet.eval()
    # loss_x = 0
    correct_train = 0
    for i, (samples, labels) in enumerate(train_data_loader):
        with torch.no_grad():
            samplesV = Variable(samples.cuda())
            labels = labels.squeeze()
            labelsV = Variable(labels.cuda())
            # labelsV = labelsV.view(-1)

            predict_label = msresnet(samplesV)
            prediction = predict_label[0].data.max(1)[1]
            correct_train += prediction.eq(labelsV.data.long()).sum()

            loss = criterion(predict_label[0], labelsV)
            # loss_x += loss.item()

    print("Training accuracy:", (100*float(correct_train)/num_train_instances))

    # train_loss[epoch] = loss_x / num_train_instances
    train_acc[epoch] = 100*float(correct_train)/num_train_instances

    trainacc = str(100*float(correct_train)/num_train_instances)[0:6]


    loss_x = 0
    correct_test = 0
    for i, (samples, labels) in enumerate(test_data_loader):
        with torch.no_grad():
            samplesV = Variable(samples.cuda())
            labels = labels.squeeze()
            labelsV = Variable(labels.cuda())
            # labelsV = labelsV.view(-1)

        predict_label = msresnet(samplesV)
        prediction = predict_label[0].data.max(1)[1]
        correct_test += prediction.eq(labelsV.data.long()).sum()

        loss = criterion(predict_label[0], labelsV)
        loss_x += loss.item()

    print("Test accuracy:", (100 * float(correct_test) / num_test_instances))

    test_loss[epoch] = loss_x / num_test_instances
    test_acc[epoch] = 100 * float(correct_test) / num_test_instances

    testacc = str(100 * float(correct_test) / num_test_instances)[0:6]

    if epoch == 0:
        temp_test = correct_test
        temp_train = correct_train
    elif correct_test>temp_test:
        torch.save(msresnet, 'weights/changingResnet/ChaningSpeed_Train' + trainacc + 'Test' + testacc + '.pkl')
        temp_test = correct_test
        temp_train = correct_train

sio.savemat('result/changingResnet/TrainLoss_' + 'ChangingSpeed_Train' + str(100*float(temp_train)/num_train_instances)[0:6] + 'Test' + str(100*float(temp_test)/num_test_instances)[0:6] + '.mat', {'train_loss': train_loss})
sio.savemat('result/changingResnet/TestLoss_' + 'ChangingSpeed_Train' + str(100*float(temp_train)/num_train_instances)[0:6] + 'Test' + str(100*float(temp_test)/num_test_instances)[0:6] + '.mat', {'test_loss': test_loss})
sio.savemat('result/changingResnet/TrainAccuracy_' + 'ChangingSpeed_Train' + str(100*float(temp_train)/num_train_instances)[0:6] + 'Test' + str(100*float(temp_test)/num_test_instances)[0:6] + '.mat', {'train_acc': train_acc})
sio.savemat('result/changingResnet/TestAccuracy_' + 'ChangingSpeed_Train' + str(100*float(temp_train)/num_train_instances)[0:6] + 'Test' + str(100*float(temp_test)/num_test_instances)[0:6] + '.mat', {'test_acc': test_acc})
print(str(100*float(temp_test)/num_test_instances)[0:6])

plt.plot(train_loss)
plt.show()
