#====== Libraries ======#
import torch.nn as nn
from torchvision import transforms
import torch
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt

# Define Model (for pickle import)
class Model(nn.Module):

        def __init__(self):
            super(Model, self).__init__()
            self.l1 = nn.Linear(784, 520)
            self.l2 = nn.Linear(520, 320)
            self.l3 = nn.Linear(320, 240)
            self.l4 = nn.Linear(240, 120)
            self.l5 = nn.Linear(120, 10)

        def forward(self, x):
            x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = F.relu(self.l3(x))
            x = F.relu(self.l4(x))
            return self.l5(x)

def predict(img, weights_dir):
    try:
        trans = transforms.ToTensor()
        model = Model()
        model.load_state_dict(torch.load(weights_dir), strict= False)
        # strict=False; so the dict is loaded correctly despite .module labelling from the nn.Sequential() structure
        output = model(trans(img))
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
