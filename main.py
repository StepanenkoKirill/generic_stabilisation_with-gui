import sys, time
from PyQt5 import QtWidgets, QtCore, uic
from GUI.ui import Ui_Form

from Stabilisation.Stabilizatoin import main

class Main(QtWidgets.QWidget, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        # uic.loadUi(r'C:\Users\Ольга\PycharmProjects\Qt5Projects\generic_stabilisation_with_gui\GUI\generic_stabilisation_with_gui3.ui', self)
        self.ui.pushButton_8.clicked.connect(self.text_edit_filler)
        self.ui.pushButton_9.clicked.connect(main)

        # main() # create another process or make start button
    def text_edit_filler(self):
        i = 1
        while i < 5:
            self.plainTextEdit.insertPlainText("You can write text here.\n")
            time.sleep(1)
            i += 1



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
