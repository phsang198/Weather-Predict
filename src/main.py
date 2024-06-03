###############################################################################################################################
import os
import sys

from PyQt6 import QtCore, QtGui, QtWidgets, uic
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import *
from PyQt6.QtWidgets import QMessageBox
from PyQt6.uic import loadUi

#import requests
from predict import predict

# Xác định đường dẫn thư mục
base_dir = os.path.dirname(os.path.abspath(__file__))
weather_dir = os.path.dirname(base_dir)
images_dir = os.path.join(weather_dir, 'images')
ui_dir = os.path.join(weather_dir, 'ui')

#cửa sổ main
class Weather_w(QMainWindow):
    def __init__(self):
        super(Weather_w,self).__init__()
        uic.loadUi(ui_dir + '/weather.ui',self)
        self.predict.clicked.connect(self.m_predict)

    def m_predict(self):
        outlook = self.outlook.currentText()
        temperature = self.temperature.text()
        humidity = self.humidity.text()
        windy = self.windy.isChecked()

        if not temperature or not humidity:
            self.show_warning_messagebox("temperature or humidity is null")
            return

        result = predict(outlook, temperature, humidity, windy)
        
        if result == 'yes':
            pixmap = QPixmap(images_dir + "/outside.png")
            self.result.setPixmap(pixmap)
            self.result.setScaledContents(True) 
        else:
            pixmap = QPixmap(images_dir + "/athome.jpg")
            self.result.setPixmap(pixmap)
            self.result.setScaledContents(True) 

    def show_warning_messagebox(self, message): 
        msg = QMessageBox() 
        msg.setText(message) 
        msg.setWindowTitle("Warning") 
        retval = msg.exec()
#xử lí
app = QApplication(sys.argv) 
widget = QtWidgets.QStackedWidget() 
Weather_f = Weather_w()
widget.addWidget(Weather_f)
widget.setCurrentIndex(0)
widget.setWindowTitle("Weather Prediction")
widget.setFixedHeight(631)
widget.setFixedWidth(620)
widget.show()
app.exec() 
