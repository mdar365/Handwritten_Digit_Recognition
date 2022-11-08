# Callum McDowell, Mazen Darwish
# March/April 2021

#====== main.py ======#
# HELPER

# IN: Void.
# OUT: Launches application.
#
# CS302 PyTorch NN Handwritten Digit Recogniser
# Repo here: https://github.com/COMPSYS-302-2021/project-1-team_04


#====== Libraries ======#
import sys;
import PyQt5.QtWidgets as qt;
import mainUI;

#====== Main() ======#
if __name__ == '__main__':
    app = qt.QApplication(sys.argv)
    appwin = mainUI.AppMainWindow()
    sys.exit(app.exec_())