import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np

#====== Hyper Parameters ======#
number_of_epochs = 4; # arbitrary
batch_size = 64;        # arbitrary
learning_rate = 1e-2;
momentum = 0.9;
k_size = 3;      # could try decreasing it

# Define Model (for pickle import)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__();
        # Input size 1x28x28 (from mono MNIST image)
        self.conv1 = nn.Conv2d(1, 4, kernel_size= k_size, stride=1, padding=1, bias=False);
        self.bn1 = nn.BatchNorm2d(4);
        # Input size 4x28x28
        self.conv2 = nn.Conv2d(4, 10, kernel_size= k_size, stride=1, padding=1, bias=False);
        self.bn2 = nn.BatchNorm2d(10);

        # Maxpool with kernel of 2 to 'halve' size
        # Input is 10x28x28, outputs 10x14x14
        self.mp1 = nn.MaxPool2d(kernel_size= 2, stride= 2);

        # Input size 10x14x14
        self.lin1 = nn.Linear(10*14*14, 12*12);
        self.lin2 = nn.Linear(12*12, 6*6);
        #self.lin3 = nn.Linear(6*6, 10);

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)));
        #print(f"\nconv1:\n{x.size()}")
        x = F.relu(self.bn2(self.conv2(x)));
        #print(f"\nconv2:\n{x.size()}")

        x = self.mp1(x);
        x = x.view(-1, 10*14*14);
        #print(f"\nview:\n{x.size()}")

        x = F.relu(self.lin1(x));
        x = F.relu(self.lin2(x));
        #x = F.relu(self.lin3(x));
        return x;


def predict(img, weights_dir):
    try:
        trans = transforms.ToTensor()
        img = trans(img);
        img = torch.unsqueeze(img, 0);  # convert from 3D to 4D (to add missing batch dimension)

        model = Model()
        model.load_state_dict(torch.load(weights_dir), strict= False)
        # strict=False; so the dict is loaded correctly despite .module labelling from the nn.Sequential() structure
        output = model(img)
        pred = output.data.max(1, keepdim=True)[1]

        #Getting the relative probability of the predictions
        relative_probability = output[0].tolist()
        if min(relative_probability) < 0:
            for value in relative_probability:
                ind = relative_probability.index(value)
                relative_probability[ind] = value + (-(min(output[0].tolist())))

        return int(pred), relative_probability;
    except Exception as e:
        print(e);
