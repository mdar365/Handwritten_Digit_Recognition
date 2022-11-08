# Mazen Darwish, Callum McDowell
# March/April 2021

#====== maincUI.py ======#
# CONTROLLER + VIEW

# IN: Owning QMainWindow.
# OUT: Main GUI content and commands (incl. canvas, requesting prediction, etc).
#
# The core content and 'central widget' of our app.
# Contains the canvas, canvas tools, model selection, model weights selection,
# prediction button, prediction result plot, and other minor features.


#====== Libraries ======#
from PyQt5.QtWidgets import *;
from PyQt5.QtCore import * 
from PyQt5.QtGui import *;

import resources as r;
import peripheralUI;
import sys
import canvasToMNIST
import ModelManager
import matplotlib.pyplot as plt

import cv2
import os
import os.path
from os import path
from PIL import Image
import matplotlib.pyplot as plt

#====== Drawing Canvas ======#
class Canvas(QWidget):
    # The widget we use to hand draw new numbers for validation/demo.
    # Is an important part of the main content window, and is added/
    # removed when the model is ready/unready to be used.

    def __init__(self):
        super().__init__();

        self.setFixedSize(600, 800)
  
        #Creating image object
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)


        #drawing flag
        self.drawing = False
        self.drawing_allowed = False;
        #brush size (ideal brush size to maximize image quality after processing)
        self.brushSize = 20
        #color
        self.brushColor = Qt.black
  
        #QPoint object to track the point
        self.lastPoint = QPoint()

    # Enable or disable drawing to the canvas (master control)
    def setDrawingAllowed(self, tf):
        if isinstance(tf, bool):
            self.drawing_allowed = tf;

    #This method checks for mouse clicks
    def mousePressEvent(self, event):
  
        #Check if left mouse button is pressed
        if event.button() == Qt.LeftButton:
            #Make drawing flag true
            self.drawing = True
            #Make the last point at the mouse cursor position
            self.lastPoint = event.pos()
  
    #This method tracks mouse activity
    def mouseMoveEvent(self, event):
          
        #Checking if left button is pressed and drawing flag is true
        if (event.buttons() & Qt.LeftButton) & self.drawing & self.drawing_allowed:
              
            #Creating painter object
            painter = QPainter(self.image)
              
            #Set the pen of the painter
            painter.setPen(QPen(self.brushColor, self.brushSize, 
                            Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
              
            #Draw line from the last point of cursor to the current point
            painter.drawLine(self.lastPoint, event.pos())
              
            #Change the last point
            self.lastPoint = event.pos()
            #Update
            self.update()
  
    #Method called when mouse button is released
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
  
    #Paint event
    def paintEvent(self, event):
        #Create a canvas
        canvasPainter = QPainter(self)
          
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
  
    #Method for clearing everything on canvas
    def clear(self):
        # make the whole canvas white
        self.image.fill(Qt.white)
        # update
        self.update()

    def submit(self):
        #Saving QImage to a file first as opencv cannot process QImage
        self.image.save("input.png")

        #Reading image with opencv
        img = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)

        #Removing file to save storage space
        os.remove("input.png")

        #Converting image to MNIST format
        img = canvasToMNIST.cropInput(img)
        img = canvasToMNIST.convertToMNIST(img)
        return img


#====== Main Content ======#
class AppMainContent(QWidget):
    # Our 'central widget' for the MainWindow frame.
    # Core content goes here.
    def __init__(self, model=None):
        super().__init__();
        
        self.model = model
        self.model_manager = ModelManager.ModelManager()

        # hbox: [ canvas, sidebox: [vbox: [...]] ]

        # | hbox:   |  sidebox: |
        # |         | --------- |
        # | canvas  |  model    |
        # |         | --------- |
        # |         |  tools    |

        self.hbox = QHBoxLayout();
        self.vbox = QVBoxLayout();
        self.canvas = Canvas();
        self.canvasbox = QWidget(self);
        self.sidebox = QWidget(self);

        self.setLayout(self.hbox);
        self.hbox.addWidget(self.canvasbox);
        self.hbox.addWidget(self.sidebox);

        # canvasBox
        self.canvasvbox = QVBoxLayout();
        self.canvasbox.setLayout(self.canvasvbox);
        self.canvasvbox.addWidget(self.canvas);
        #self.canvasvbox.addStretch(1);

        # sideBox
        self.sidebox.setLayout(self.vbox);
        
        # modelGroupBox
        self.modelGroup = QGroupBox("Model Options");
        self.modelGroupLayout = QVBoxLayout();
        self.modelGroup.setLayout(self.modelGroupLayout);
        self.vbox.addWidget(self.modelGroup);
        # -- modelLabel
        self.modelLabel = QLabel();
        self.modelLabel.setFont(QFont('Sans Serif', 10));
        self.updateModelLabel();
        self.modelGroupLayout.addWidget(self.modelLabel);
        # -- modelSelectCombo
        self.modelSelectCombo = QComboBox();
        self.modelSelectCombo.addItems(ModelManager.MODEL_LIST);
        self.modelSelectCombo.textActivated.connect(lambda comboEvent: self.setModelName(self.modelSelectCombo.currentText()));
        self.modelGroupLayout.addWidget(self.modelSelectCombo);
        # -- modelWeightButton
        self.modelWeightButton = QPushButton("Model Weights", self);
        self.modelWeightButton.clicked.connect(self.changeModelWeights);
        self.modelGroupLayout.addWidget(self.modelWeightButton);

        self.vbox.addStretch(1);

        # toolsGroupBox
        self.toolsGroup = QGroupBox("Tools");
        self.toolsGroupLayout = QVBoxLayout();
        self.toolsGroup.setLayout(self.toolsGroupLayout);
        self.vbox.addWidget(self.toolsGroup);
        # -- clearButton
        self.clearButton = QPushButton("Clear", self);
        self.clearButton.clicked.connect(self.clear);
        self.toolsGroupLayout.addWidget(self.clearButton);
        # -- predictionLabel
        self.predLabel = QLabel(self);
        self.predLabel.setFont(QFont('Sans Serif', 10));
        self.setPredLabel("");
        self.toolsGroupLayout.addWidget(self.predLabel);
        # -- submitButton
        self.submitButton = QPushButton("Submit", self);
        self.submitButton.clicked.connect(self.submit);
        self.toolsGroupLayout.addWidget(self.submitButton);
        # -- showGraphButton
        self.toggleGraphButton =  QPushButton('Hide/Show Graph', self);
        self.toggleGraphButton.hide();
        self.toggleGraphButton.clicked.connect(self.togglePlot);
        self.toolsGroupLayout.addWidget(self.toggleGraphButton);
        self.plotPixmap = QLabel(self)
        self.toolsGroupLayout.addWidget(self.plotPixmap);

        self.vbox.addStretch(5);

    def setModelName(self, text):
        self.model_manager.setModelName(text);
        self.updateModelLabel();

    def updateModelLabel(self):
        self.modelLabel.setText("Model: " + self.model_manager.model_name);

    def setPredLabel(self, text):
        self.predLabel.setText('Predicted value is: <b><span style="font-size:18px";font="monospace">' + text + "</b></span>");

    def submit(self):
        #Exception would be executed if no input is found
        try:
            img = self.canvas.submit();
            self.updateModelLabel();
        except:
            self.generateErrorBox("Error", "No canvas input to submit");
            return;

        try:
            pred, self.plot = self.model_manager.predictWithModel(img);
            self.setPredLabel(str(pred));
            self.toggleGraphButton.show();
            self.plotPixmap.setPixmap(QPixmap("probability_graph.png"))
            os.remove("probability_graph.png")
        except:
            # None is returned if predict() fails.
            self.changeModelWeights();

    def clear(self):
        #Close plot if it's still open
        try:
            plt.close();
        except Exception:
            pass;
        finally:
            self.setPredLabel("");
            self.canvas.clear();

    #Toggle probability graph when the "Hide/Show graph" button is clicked
    def togglePlot(self):
        if (self.plotPixmap.isVisible()):
            self.plotPixmap.hide()
        else:
            self.plotPixmap.show();

    def changeModelWeights(self):
        self.model_manager.changeModelWeightsDir(self);

    def getModelManager(self):
        return self.model_manager;

    def generateErrorBox(self, title="Error", message="Error", detail="None"):
        error_box = peripheralUI.ErrorBox(title, message, detail);
        error_box.render();

  
        

