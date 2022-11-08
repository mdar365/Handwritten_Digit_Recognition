# Callum McDowell, Mazen Darwish
# March/April 2021

#====== mainUI.py ======#
# VIEW + CONTROLLER

# IN: Void.
# OUT: CentralWidget, and various utility commands for our app
#
# The MainWindow of our app.
# The MainWindow provides a framework for the UI, with our QToolBar,
# QMenuBar, and QStatusBar implemented here. Most actions and
# shortcuts are also initialised here.
# A QMainWindow *must* have a 'central widget' to display, which holds
# the our app's content. Our central widget is under 'maincUI.py'.
# https://doc.qt.io/qt-5/qmainwindow.html


#====== Libraries ======#
import sys;
from PyQt5.QtWidgets import *;
from PyQt5.QtGui import QIcon;

import time
import resources as r;
import maincUI as c;
import peripheralUI;
import ModelManager;



#====== Window Setup ======#
# See file documentation for functionality.
# PyQt5 guide here for components: https://realpython.com/python-menus-toolbars/
class AppMainWindow(QMainWindow):
    main_content = 0;   # allow us to access central widget instance from within all class methods

    def __init__(self):
        super().__init__();
        self.initUI();

    def initUI(self):
        # Defines
        WINDOW_SIZE_X = 1000;
        WINDOW_SIZE_Y = 800;
        WINDOW_TITLE = "Handwritten Digit Recognizer";
        self.layout = QVBoxLayout
        # Code
        self.setWindowTitle(WINDOW_TITLE);
        self.resize(WINDOW_SIZE_X, WINDOW_SIZE_Y);
        #self.setFixedSize(WINDOW_SIZE_X, WINDOW_SIZE_Y);
        self.setWindowIcon(QIcon(r.ICON_WINDOW));
        self.centreWindow();
        # Core Components
        self.initActions();
        self.menubar = self.initMenuBar();
        self.toolbar = self.initMainToolBar();
        self.statusbar = self.initStatusBar();

        AppMainWindow.main_content = c.AppMainContent();
        self.setCentralWidget(AppMainWindow.main_content);

        self.show();

    def centreWindow(self):
        rectangle_frame = self.frameGeometry();
        centre_point = QDesktopWidget().availableGeometry().center();
        rectangle_frame.moveCenter(centre_point);
        self.move(rectangle_frame.topLeft());

    def initActions(self):
        # Actions defined here are owned by AppMainWindow and persist
        # Exit
        self.exitAction = QAction("&Exit", self);
        self.exitAction.setIcon(QIcon(r.ICON_EXIT));
        self.exitAction.setShortcut("Ctrl+E");
        self.exitAction.triggered.connect(self.exitApp);
        # Train Model
        self.trainModel = QAction("&Train Model", self);
        self.trainModel.setIcon(QIcon(r.ICON_WORKING));
        self.trainModel.setToolTip("Train handwriting recognition model");
        self.trainModel.setStatusTip("Train handwriting recognition model");
        self.trainModel.triggered.connect(self.modelTraining);
        # Help 
        self.helpAction = QAction("Help", self);
        self.helpAction.setIcon(QIcon(r.ICON_HELP));
        self.helpAction.triggered.connect(self.helpDialogue);
        # Draw
        self.drawTool = QToolButton(self);
        self.drawTool.setIcon(QIcon(r.ICON_DRAW));
        self.drawTool.setToolTip("Toggle drawing on the canvas");
        self.drawTool.setCheckable(True);
        self.drawTool.setShortcut("Ctrl+D");
        self.drawTool.toggled.connect(self.startDrawing);
        # Clear
        self.clearAction = QAction("&Clear", self);
        self.clearAction.setIcon(QIcon(r.ICON_TRASH));
        self.clearAction.setToolTip("Clear the canvas");
        self.clearAction.setShortcut("Ctrl+F");
        self.clearAction.triggered.connect(self.clearDrawing);
        # View
        self.viewTrainingImagesAction = QAction("View Training Images", self);
        self.viewTrainingImagesAction.setIcon(QIcon(r.ICON_FIND));
        self.viewTrainingImagesAction.triggered.connect(self.viewTrainingImages)
        self.viewTestingImagesAction = QAction("View Testing Images", self);
        self.viewTestingImagesAction.setIcon(QIcon(r.ICON_FIND));
        self.viewTestingImagesAction.triggered.connect(self.viewTestingImages)

        # Note: Add actions to context menus for drawing canvas
        # https://realpython.com/python-menus-toolbars/#creating-context-menus-through-context-menu-policy


    def initMenuBar(self):
        self.menubar = self.menuBar();

        self.fileMenu = self.menubar.addMenu("&File");
        self.fileMenu.addAction(self.trainModel);
        self.fileMenu.addAction(self.helpAction);
        self.fileMenu.addSeparator();
        self.fileMenu.addAction(self.exitAction);

        self.viewMenu = self.menubar.addMenu("&View");
        self.viewMenu.addAction(self.viewTrainingImagesAction);
        self.viewMenu.addAction(self.viewTestingImagesAction);

        #submenu = menu.addMenu("name");
        #submenu.addAction(...);

        return self.menubar;

    def initMainToolBar(self):
        self.toolbar = self.addToolBar("Main Tools");
        self.toolbar.addSeparator();
        
        self.toolbar.addWidget(self.drawTool);
        self.toolbar.addAction(self.clearAction);

        return self.toolbar;

    def initStatusBar(self):
        self.statusbar = self.statusBar();
        self.statusbar.showMessage("Ready");

        #self.formatted_label = QLabel(f"{self.func()} text");
        #self.statusbar.addPermanentWidget(...);
        return self.statusbar;

    def helpDialogue(self):
        self.popup = peripheralUI.PopupBox("Help", r.ICON_HELP);
        self.popup.assignText("By: Mazen Darwish, Callum McDowell")
        self.popup.assignText('Full documentation at: <a href="https://github.com/COMPSYS-302-2021/project-1-team_04">https://github.com/COMPSYS-302-2021/project-1-team_04</a>');
        self.popup.assignText("");  # new line (\n doesn't work)
        self.popup.assignText("<b>Icon credit to:</b>");
        self.popup.assignText('Icons used with permission from <a href="https://www.freepik.com">Freepik</a> from <a href="https://www.flaticon.com/">www.flaticon.com</a>.');
        self.popup.assignText('Icons used with permission from <a href="https://iconmonstr.com">iconmonstr</a>.');

    def modelTraining(self):
        try:
            train_dlg = ModelManager.ModelDialog(parent= self, manager= self.centralWidget().model_manager);
        except Exception as e:
            print(e);
            return;

    def startDrawing(self):
        try:
            if (AppMainWindow.main_content.canvas.drawing_allowed == True):
                AppMainWindow.main_content.canvas.drawing_allowed = False;
            else:
                AppMainWindow.main_content.canvas.drawing_allowed = True;
        except ValueError:
            return;
            
    def clearDrawing(self):
        try:
            AppMainWindow.main_content.clear();
        except ValueError:
            return;

    def viewTrainingImages(self):
        viewerDlg = peripheralUI.ViewImagesDlg("training")
        viewerDlg.exec_()

    def viewTestingImages(self):
        viewerDlg = peripheralUI.ViewImagesDlg("testing")
        viewerDlg.exec_()

    def exitApp(self):
        confirm = QMessageBox.question(self, "Warning", "Are you sure you want to quit?",
                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No);
        if (confirm == QMessageBox.Yes):
          self.close();