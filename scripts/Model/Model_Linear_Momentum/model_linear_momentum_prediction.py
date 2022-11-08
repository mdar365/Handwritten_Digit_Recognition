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
        l6 = 10*10;
        l7 = 10;

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
            nn.ReLU(),
            nn.Linear(l6, l7),
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


#====== Prediction, Recall, and F1 metrics version of test() ======#
# def test(self, dataloader, model, loss_fn):
#         size = len(dataloader.dataset);
#         test_loss, correct = 0, 0;
        
#         # Lists of 10x1 arrays, where labels are used to index.
#         true_positive = [0] * 10;
#         false_positive = [0] * 10;
#         false_negative = [0] * 10;
#         sum_tp = [0] * 10;
#         sum_fp = [0] * 10;
#         sum_fn = [0] * 10;

#         with torch.no_grad():       # disable learning; no back propagation
#             for X, y in dataloader:
#                 pred = model(X);
#                 test_loss += loss_fn(pred, y).item();
#                 correct += (pred.argmax(1) == y).type(torch.float).sum().item();
                
#                 #== Precision and Recall Metrics ==#                
#                 # Note that y is a tensor of the labels
#                 # pred.argmax(1) is a 1D tensor of the predictions made this minibatch
#                 # (pred.argmax(1) == y) is a bool mask that is 1 where the prediction is correct
#                 pred_1D = pred.argmax(1);
#                 for index, l in enumerate(pred_1D):
#                     if (l == y[index]): # if (prediction correct):
#                       true_positive[y[index].item()] += 1;
#                     else:
#                       false_negative[y[index].item()] += 1;
#                       false_positive[l.item()] += 1;

#                 for i in range(10):
#                     try:
#                       prec = (true_positive[i] / (true_positive[i] + false_positive[i]) );
#                     except ZeroDivisionError:
#                       pass;
#                     try:
#                       rec = (true_positive[i] / (true_positive[i] + false_negative[i]) );
#                     except ZeroDivisionError:
#                       pass;
#                     # print(f"Precision[{i}] = { prec * 100 :>0.2f}%");
#                     # print(f"Recall[{i}] = { rec * 100 :>0.2f}%");
#                 # prec_total = sum(true_positive) / (sum(true_positive) + sum(false_positive));
#                 # rec_total = sum(true_positive) / (sum(true_positive) + sum(false_negative));
#                 # print(f"Precision = { prec_total * 100 :>0.2f}");
#                 # print(f"Recall = { rec_total * 100 :>0.2f}");

#                 for j in range(10):
#                     sum_tp[j] += true_positive[j];
#                     sum_fp[j] += false_positive[j];
#                     sum_fn[j] += false_negative[j];
#                 true_positive = [0] * 10;
#                 false_positive = [0] * 10;
#                 false_negative = [0] * 10;
            
#             test_loss /= size;      # equivalent to test_loss = test_loss/size;
#             correct /= size;
#             accuracy = 100*correct;
#             print (f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n");

#             prec_total = sum(sum_tp) / (sum(sum_tp) + sum(sum_fp));
#             rec_total = sum(sum_tp) / (sum(sum_tp) + sum(sum_fn));
#             print(f"Total Precision = { prec_total * 100 :>0.2f}");
#             print(f"Total Recall = { rec_total * 100 :>0.2f}");

#             print("\nPrecision by class:");
#             for j in range(10):
#             try:
#                 print(f"{j} | {sum_tp[j]/(sum_tp[j] + sum_fp[j])}");
#             except ZeroDivisionError:
#                 print(f"{j} | missing");
#             print("\nRecall by class:");
#             for j in range(10):
#             try:
#                 print(f"{j} | {sum_tp[j]/(sum_tp[j] + sum_fn[j])}");
#             except ZeroDivisionError:
#                 print(f"{j} | missing");

#         return accuracy;