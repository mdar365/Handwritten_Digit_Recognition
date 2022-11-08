# Callum McDowell, Mazen Darwish
# April 2021

#====== ModelManager.py ======#
# CONTROLLER

# IN: Requests to train and use models.
# OUT: Prediction results, GUI notifications.
#
# Trains the models and selects which one to use.
# All model and prediction requests must be sent to
# this module.

#====== Libraries ======#
import resources as r
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from torchvision import datasets, transforms;
import matplotlib.pyplot as plt
import numpy as np
import peripheralUI
import pandas as pd
import os
from urllib.error import HTTPError
import threadsafe
import resources as r

# Model Linear
import Model.Model_Linear.model_linear as model_linear
import Model.Model_Linear.model_linear_prediction as model_linear_prediction
# Model Linear Momentum
import Model.Model_Linear_Momentum.model_linear_momentum as model_linear_momentum
import Model.Model_Linear_Momentum.model_linear_momentum_prediction as model_linear_momentum_prediction
# Model Convolutional
import Model.Model_Conv.model_conv as model_conv
import Model.Model_Conv.model_conv_prediction as model_conv_prediction
# Model Original
import Model.Model_Original.model_original as model_original
import Model.Model_Original.model_original_prediction as model_original_prediction
# ...
MODEL_LIST = ["Linear Momentum", "Linear", "Convolutional", "Original"]
# Note: default to Linear Momentum


# ModelManager
# Interface for requests to the model
# Use it to select the current model, train

class ModelManager():
    def __init__(self):
        self.model_details = pd.read_json(r.MODEL_CONFIG_DIR)["models"]
        self.model_name = MODEL_LIST[0];
        self.model_weights_folder =  r.MODULE_DIR + "/Model/"+self.model_details[self.model_name];  # must be initialised as is base for QFileDialog
        
        for filename in os.listdir(self.model_weights_folder):
            if filename.endswith("_weights.pkl"):
                self.model_weights_file = self.model_weights_folder+ "/"+filename;
        self.plot_probabilities = None;

    def setModelName(self, name : str):
        if (isinstance(name, str)):
            self.model_name = name;
            self.model_weights_folder = r.MODULE_DIR + "/Model/"+self.model_details[self.model_name];
            for filename in os.listdir(self.model_weights_folder):
                if filename.endswith("_weights.pkl"):
                    self.model_weights_file = self.model_weights_folder+ "/"+filename;

    def changeModelWeightsDir(self, owner):
        # Owner is QWidget to act as parent
        weights_dir, _ = QFileDialog.getOpenFileName(owner,"Please select model weights", self.model_weights_file ,"pickle files (*.pkl)")
        if (len(weights_dir) > 0):
            self.model_weights_file = weights_dir;

    def predictWithModel(self, image):
        try:
            if (self.model_name == "Convolutional"):
                pred, self.plot_probabilities = model_conv_prediction.predict(image, self.model_weights_file);

            elif (self.model_name == "Original"):
                pred, self.plot_probabilities = model_original_prediction.predict(image, self.model_weights_file);

            elif (self.model_name == "Linear Momentum"):
                pred, self.plot_probabilities = model_linear_momentum_prediction.predict(image, self.model_weights_file);
                
            else: # Linear
                pred, self.plot_probabilities = model_linear_prediction.predict(image, self.model_weights_file);
                
            plot = self.createBarPlot();    
            return pred, plot;
        except Exception as e:
            # If an invalid file is loaded...
            self.generateErrorBox("Error", "Invalid Model", e)
            return;

    def createBarPlot(self):
        plot = self.plot_bar(self.plot_probabilities);
        plt.savefig("probability_graph.png")

        mngr = plt.get_current_fig_manager();
        mngr.window.setGeometry(50,100, 600,600);
        return plot;

    def plot_bar(self, probability):
        plt.close() # Close previous plot if it's still open

        # Crop classification option for 0->9 (if convolutional model is configured incorrectly)
        probability = probability[:10] 
        # Normalise to 1 to get % values
        temp = [(100 * float(i))/sum(probability) for i in probability];
        probability = temp;

        # Get array of indices 
        index = np.arange(len(probability)) 
        # Plot index on x-axis and probability on y-axis
        plot = plt.bar(index, probability)

        #Add labels
        plt.xlabel('Digit', fontsize=15)
        plt.ylabel('Probability', fontsize=20)
        plt.xticks(index, fontsize=8, rotation=30)
        plt.title('Model Prediction Probability')
        return plot;

    def generateErrorBox(self, title="Error", message="Error", detail="None"):
        error_box = peripheralUI.ErrorBox(title, message, detail);
        error_box.render();



# Download and Training Dialogue
# Allows the user to download the MNIST dataset and train the model.
# Shows progress downloading and training with a progress bar.

class ModelDialog(QDialog):
    def __init__(self, parent=None, manager=None):
        # parent is the QWidget that will own the dialog
        # manager is the ModelManager() instance that stores the model data
        super().__init__(parent=parent)
        self.model_manager = manager;
        self.threadpool = threadsafe.QThreadPool();

        self.setWindowTitle("Train Model")
        self.setWindowIcon(QIcon(r.ICON_WORKING))
        
        self.textBox = QTextEdit()
        self.textBox.setReadOnly(True)

        self.progressBar = QProgressBar()

        self.downloadDataButton = QPushButton("&Download Dataset", self);
        self.downloadDataButton.setToolTip("Download MNIST Dataset");
        self.downloadDataButton.setStatusTip("Download MNIST Dataset");
        self.downloadDataButton.clicked.connect(self.downloadMNISTData);

        self.modelCombo = QComboBox();
        self.modelCombo.setToolTip("Select your model structure")
        self.modelCombo.addItems(MODEL_LIST);

        self.trainButton = QPushButton("&Train Model", self);
        self.trainButton.setToolTip("Train Model");
        self.trainButton.setStatusTip("Train Model");
        self.trainButton.clicked.connect(lambda startTrain: self.setAndTrainModel(self.modelCombo.currentText()));
        # We must register an interim function ('startTrain()') for the event to call a func with params correctly
        # https://forum.qt.io/topic/60640/pyqt-immediately-calling-function-handler-upon-setup-why/4

        self.cancelButton = QPushButton("&Close", self);
        self.cancelButton.setToolTip("Close model-creation window");
        self.cancelButton.setStatusTip("Close model-creation window");
        self.cancelButton.clicked.connect(self.close)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.textBox)
        self.layout.addWidget(self.progressBar)
        self.layout.addWidget(self.downloadDataButton)
        self.layout.addWidget(self.modelCombo)
        self.layout.addWidget(self.trainButton)
        self.layout.addWidget(self.cancelButton)
        self.setLayout(self.layout)

        self.show();

    def newThreadWorker(self, fn, *args):
        # Execute function in a different thread
        worker = threadsafe.Worker(fn,args);
        self.threadpool.start(worker);

    def pureDownload(self, b_is_train : bool):
        # Multithread safe
        if (isinstance(b_is_train, bool)):
            if (b_is_train == True):
                root_dir = r.DATASET_DIR + "trainset"
            else:
                root_dir = r.DATASET_DIR + "testset"

            datasets.MNIST(
                root=root_dir,
                train= b_is_train,
                download= True,
                transform= transforms.ToTensor()
            )     

    def downloadMNISTData(self):
        self.textBox.append("Downloading dataset...")
        self.textBox.repaint()
        # Downloading MNIST Dataset (if it doesn't already exist)
        try:
            self.newThreadWorker(self.pureDownload,True)
            self.textBox.append("Dataset already downloaded!")
            self.progressBar.setValue(50)
        except HTTPError as err:
            if err.code == 503:
                self.textBox.append("HTTP Error 503: Service Unavailable")
        except:
            self.newThreadWorker(self.pureDownload,False)
            self.textBox.append("Dataset downloaded!")

    def setAndTrainModel(self, model_str):
        self.model_manager.setModelName(model_str);
        self.trainModel(model_str)

    def trainModel(self, model_str):
        self.textBox.append(f"Training {model_str} model...");
        try:
            if (model_str == "Convolutional"):
                x = model_conv.modelTrainingFramework();
                self.accuracy = x.trainModel(self.progressBar);

            elif (model_str == "Original"):
                x = model_original.modelTrainingFramework();
                self.accuracy = x.trainModel(self.progressBar);

            elif (model_str == "Linear Momentum"):
                x = model_linear_momentum.modelTrainingFramework();
                self.accuracy = x.trainModel(self.progressBar);

            else:
                # default to linear model
                x = model_linear.modelTrainingFramework();
                self.accuracy = x.trainModel(self.progressBar);

        except Exception as e:
            self.textBox.append("Error training the model. Make sure the model has been downloaded first by pressing the 'Download Dataset' button");
            print(e);
        else:
            self.textBox.append(f"Training Done\nAccuracy: {self.accuracy :>.2f}%");

    def newThreadWorker(self, fn, *args):
        # Execute function in a different thread
        worker = threadsafe.Worker(fn,args);
        self.threadpool.start(worker);
