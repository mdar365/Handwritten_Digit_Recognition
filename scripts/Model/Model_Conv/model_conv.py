# First custom model.
# [TODO]: Explain further, add timer, tweak: stacks + loss fn + optimiser
# https://pytorch.org/tutorials/beginner/basics/intro.html


#====== Libraries ======#
import resources as r
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os;
import torch;
import torchvision;
import time;
# Data:
from torch.utils import data;
from torchvision import datasets, models;
from torchvision.transforms import ToTensor, Lambda;
from torch.nn import functional as F
# Model:
from torch import nn, cuda;

SAVE_DIR = r.MODULE_DIR + "/Model/Model_Conv/"
MODEL_CODE = "Conv"


#====== Hyper Parameters ======#
number_of_epochs = 4; # arbitrary
batch_size = 64;        # arbitrary
learning_rate = 1e-2;
momentum = 0.9;
k_size = 3;      # could try decreasing it

#====== Model ======#
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


# trainModel()
# 

class modelTrainingFramework():
    # Wrapper class so that functions can be called on instantiated object
    def trainModel(self, progress_bar):
        #====== Datasets ======#
        trainset = datasets.MNIST(
            root=r.DATASET_DIR + "trainset",
            train= True,
            download= True,
            transform= ToTensor()
        );

        loader_trainset = data.DataLoader(
            dataset= trainset,
            batch_size= batch_size,
            shuffle= True,
        );

        testset = datasets.MNIST(
            root=r.DATAST_DIR + "testset",
            train= False,
            download= True,
            transform= ToTensor()
        );

        loader_testset = data.DataLoader(
            dataset= testset,
            batch_size= batch_size,
            shuffle= False
        );

        #====== Model Instance ======#
        device = "cpu";
        if cuda.is_available():
            device = "cuda";

        self.net = Model().to(device);

        #====== Loss and Optimiser ======#
        self.loss_fn = nn.CrossEntropyLoss()
            # NOTE: try NLLLoss or CrossEntropyLoss for negative log
            #       (negative log is better for classification)
        self.optimiser = torch.optim.SGD(self.net.parameters(), lr= learning_rate);
            # NOTE: experiment with different optimisers, not just SDG

        #====== Training Epochs ======#
        print(f"Starting training with {MODEL_CODE} on device {device}\n{'=' * 24}");

        t0 = time.perf_counter()

        for i in range(number_of_epochs):
            print(f"Epoch {i+1}\n----------------------------")
            self.train(loader_trainset, self.net, self.loss_fn, self.optimiser, i, number_of_epochs, progress_bar);
            accuracy = self.test(loader_testset, self.net, self.loss_fn);

        t1 = time.perf_counter();
        progress_bar.setValue(100);
        print(f"Finished in {(t1 - t0):>.2f}s.");
        print("FIN.")

        return accuracy;



    def train(self, dataloader, model, loss_fn, optimiser, epochs_complete, epochs_total, progress_bar):
        size = len(dataloader.dataset);

        for index, (X,y) in enumerate(dataloader):
            # as we iterate over '(X,y)' 'index' (from enumerate()) tracks our progress
            
            # Forward
            pred = model(X);
            loss = loss_fn(pred, y);

            # Back propagation:
            optimiser.zero_grad();
            loss.backward();
            optimiser.step();

            if index % 100 == 0:
                loss, progress = loss.item(), index * len(X);
                print(f"loss: {loss:>7f} [{progress:>5d}/{size:>5d}]");
                # e.x. output:  "loss: 1.234567 [    0/60000]""
                
                # Export and save model
                torch.save(model, SAVE_DIR + "model_" + MODEL_CODE + ".pkl");
                torch.save(optimiser.state_dict(), SAVE_DIR + "model_" + MODEL_CODE + "_optimiser.pkl");
                torch.save(model.state_dict(), SAVE_DIR + "model_" + MODEL_CODE + "_weights.pkl");

                # Completion = ((dataset size * completed epochs) + (index / dataset size)) / (dataset size * epochs)
                # We are using ratios to keep number sizes sensible.
                completed_val = size * (epochs_complete/epochs_total);
                in_progress_val = progress * (1/epochs_total);
                completion = (completed_val + in_progress_val) / size;
                # Convert completion from normalised 0->1 to 50->100
                completion = 50 + (completion * 50);

                progress_bar.setValue(int(completion));


    def test(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset);
        test_loss, correct = 0, 0;

        with torch.no_grad():       # disable learning; no back propagation
            for X, y in dataloader:
                pred = model(X);
                test_loss += loss_fn(pred, y).item();
                correct += (pred.argmax(1) == y).type(torch.float).sum().item();
            
            test_loss /= size;      # equivalent to test_loss = test_loss/size;
            correct /= size;
            accuracy = 100*correct;
            print (f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n");

        return accuracy;
