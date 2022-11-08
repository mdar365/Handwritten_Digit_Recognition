import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np

# Define Model (for pickle import)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__();
        self.Flatten = nn.Flatten();    # Convert images from 2D to 1D array

        l1 = 26*26;
        l2 = 24*24;
        l3 = 20*20;
        l4 = 18*18;
        l5 = 16*16;
        l6 = 10;

        self.composite_stack = nn.Sequential(
            # 6 layer stack
            # linear downscaling in data size
            # ReLu to identify non-linear behaviour for closer fitting
            nn.Linear(28*28, l1),
            nn.ReLU(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, l3),
            nn.ReLU(),
            nn.Linear(l3, l4),
            nn.ReLU(),
            nn.Linear(l4, l5),
            nn.ReLU(),
            nn.Linear(l5, l6),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.Flatten(x);
        logits = self.composite_stack(x);
        return logits;


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
