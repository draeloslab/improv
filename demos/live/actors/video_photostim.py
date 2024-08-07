# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'video_tuning_curves.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 909)
        MainWindow.setStyleSheet("QMainWindow { background-color: \'blue\'; }")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("#centralwidget { background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgb(50, 50, 50), stop:1 rgba(80, 80, 80, 255)); }\n"
"#checkBox {color: black}\n"
"#label_2 {color: rgb(225, 230, 240)}\n"
"#label_3 {color: rgb(225, 230, 240)}\n"
"#label_4 {color: rgb(225, 230, 240)}\n"
"#label_5 {color: rgb(225, 230, 240)}\n"
"#frame {background-color: rgb(150, 160, 175);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px}\n"
"#frame_3 {background-color: rgb(170, 185, 200);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px}\n"
"#frame_2 {background: rgb(229, 229, 229);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 1px}\n"
"#frame_4 {background: rgb(170, 185, 200);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 1px}\n"
"#frame_5 {background: rgb(229, 229, 229);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px}\n"
"#frame_6 {background: rgb(229, 229, 229);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px}\n"
"#frame_8 {background: rgb(229, 229, 229);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px}\n"
"#frame_7 {background: rgb(229, 229, 229);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px}\n"
"#frame_9 {background: rgb(229, 229, 229);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 0px}\n"
"#frame_10 {background: rgb(229, 229, 229);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px}\n"
"#pushButton {\n"
"background-color: rgb(225, 230, 240);\n"
"color: black;\n"
"font: bold;\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px\n"
"}\n"
"#pushButton_2 {\n"
"background-color: rgb(225, 230, 240);\n"
"color: black;\n"
"font: bold;\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px\n"
"}\n"
"#pushButton_3 {\n"
"background-color: rgb(225, 230, 240);\n"
"color: black;\n"
"font: bold;\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px\n"
"}\n"
"#pushButton_4 {\n"
"background-color: rgb(200, 140, 140);\n"
"color: black;\n"
"font: bold;\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px\n"
"}")
        self.centralwidget.setObjectName("centralwidget")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(24, 308, 441, 521))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.rawplot = ImageView(self.frame_2)
        self.rawplot.setGeometry(QtCore.QRect(10, 8, 425, 507))
        self.rawplot.setObjectName("rawplot")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(14, 32, 151, 209))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 20, 111, 51))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.frame)
        self.pushButton_3.setGeometry(QtCore.QRect(20, 92, 111, 51))
        self.pushButton_3.setObjectName("pushButton_3")
        self.checkBox = QtWidgets.QCheckBox(self.frame)
        self.checkBox.setGeometry(QtCore.QRect(20, 164, 117, 41))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox.setFont(font)
        self.checkBox.setObjectName("checkBox")
        self.frame_5 = QtWidgets.QFrame(self.centralwidget)
        self.frame_5.setGeometry(QtCore.QRect(508, 308, 467, 521))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.rawplot_2 = ImageView(self.frame_5)
        self.rawplot_2.setGeometry(QtCore.QRect(10, 10, 453, 505))
        self.rawplot_2.setObjectName("rawplot_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(174, 38, 121, 73))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(186, 150, 99, 105))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(30, 286, 131, 21))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(510, 286, 201, 21))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.frame_6 = QtWidgets.QFrame(self.centralwidget)
        self.frame_6.setGeometry(QtCore.QRect(302, 22, 679, 111))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.grplot = PlotWidget(self.frame_6)
        self.grplot.setGeometry(QtCore.QRect(6, 6, 665, 101))
        self.grplot.setObjectName("grplot")
        self.frame_8 = QtWidgets.QFrame(self.centralwidget)
        self.frame_8.setGeometry(QtCore.QRect(298, 152, 681, 113))
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.grplot_2 = PlotWidget(self.frame_8)
        self.grplot_2.setGeometry(QtCore.QRect(6, 6, 667, 101))
        self.grplot_2.setObjectName("grplot_2")
        self.dir_icon = QtWidgets.QLabel(self.centralwidget)
        self.dir_icon.setGeometry(QtCore.QRect(184, 106, 67, 65))
        self.dir_icon.setText("")
        self.dir_icon.setObjectName("dir_icon")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 26))
        self.menubar.setObjectName("menubar")
        self.menuRASP_Display = QtWidgets.QMenu(self.menubar)
        self.menuRASP_Display.setObjectName("menuRASP_Display")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuRASP_Display.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_2.setText(_translate("MainWindow", "Setup"))
        self.pushButton_3.setText(_translate("MainWindow", "Run"))
        self.checkBox.setText(_translate("MainWindow", " Live Update"))
        self.label_2.setText(_translate("MainWindow", "Population\n"
"Average"))
        self.label_3.setText(_translate("MainWindow", "Selected\n"
"Neuron"))
        self.label_4.setText(_translate("MainWindow", "Raw Frame"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#eeeeec;\">Processed Frame</span></p></body></html>"))
        self.menuRASP_Display.setTitle(_translate("MainWindow", "Nexus Display"))

from pyqtgraph import ImageView, PlotWidget

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

