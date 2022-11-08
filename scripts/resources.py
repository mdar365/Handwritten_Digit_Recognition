#====== resources.py ======#
# HELPER

# A header file for consolidating and simplifying access of
# static resources (e.g. icons, graphics).
# Constants are directory paths and names.

#====== Code ======#
import os;
MODULE_DIR = os.path.dirname(os.path.realpath(__file__));
RESOURCES_DIR = MODULE_DIR + "\\resources\\";
# Set up relative DIR for referencing local filepaths
# Could replace with Qt resource system, but having issues recognising pyrcc5
# https://doc.qt.io/qt-5/resources.html


#====== Paths ======#
MODEL_CONFIG_DIR = MODULE_DIR + "\\Model\\" + "model_config.json";
DATASET_DIR = MODULE_DIR + "\\Dataset\\"
TRAINSET_DIR = DATASET_DIR + "trainset\\MNIST\\"
TESTSET_DIR = DATASET_DIR + "testset\\MNIST\\"

#====== Icons ======#
ICON_WINDOW = RESOURCES_DIR + "icon_robot.png";
ICON_EXIT = RESOURCES_DIR + "icon_exit.svg";
ICON_OPEN = RESOURCES_DIR + "icon_open.svg";
ICON_HELP = RESOURCES_DIR + "icon_help.svg";
ICON_DRAW = RESOURCES_DIR + "icon_draw.svg";
ICON_TRASH = RESOURCES_DIR + "icon_trash.svg";
ICON_WORKING = RESOURCES_DIR + "icon_working.svg";
ICON_WARNING = RESOURCES_DIR + "icon_warning.svg";
ICON_FIND = RESOURCES_DIR + "icon_find.svg";

