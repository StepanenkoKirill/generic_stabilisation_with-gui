import sys, time
from PyQt5 import QtWidgets, QtCore, uic

class Main(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(r'C:\Users\Ольга\PycharmProjects\Qt5Projects\generic_stabilisation_with_gui\GUI\generic_stabilisation_with_gui3.ui', self)
        self.pushButton_8.clicked.connect(QtWidgets.qApp.quit)

# object creation
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
