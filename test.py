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

# from model.multi_scale_ori import *
# from multi_scale_nores import *
# from multi_scale_one3x3 import *
# from multi_scale_one5x5 import *
# from multi_scale_one7x7 import *

batch_size = 1024

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

# msresnet = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=6)
msresnet = torch.load('weights/changingResnet/ChaningSpeed_Train98.655Test95.690.pkl')
msresnet = msresnet.cuda()
msresnet.eval()

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

    if i == 0:
        batch_prediction= prediction
        batch_featuremap = predict_label[1].data
        fault_prediction = batch_prediction
        featuremap = batch_featuremap

    elif i > 0:
        batch_prediction = prediction
        batch_featuremap = predict_label[1].data

        fault_prediction = np.concatenate((fault_prediction, batch_prediction), axis=0)
        featuremap = np.concatenate((featuremap, batch_featuremap), axis=0)

print("Test accuracy:", (100 * float(correct_test) / num_test_instances))


