from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
import os
from os import path
import sys
import pyqtgraph as pg
from logic_app import Logic
from pydub import AudioSegment
from pydub.playback import play
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
import vlc


FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "main.ui"))

class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.gridLayout = QGridLayout()
        self.setLayout(self.gridLayout)
        
        # Labels
        self.labelSlider1.setText("   ")
        self.labelSlider2.setText("   ")
        self.labelSlider3.setText("   ")
        self.labelSlider4.setText("   ")
        self.labelSlider5.setText("   ")
        self.labelSlider6.setText("   ")
        self.labelSlider7.setText("   ")
        self.labelSlider8.setText("   ")
        self.labelSlider9.setText("   ")
        self.labelSlider10.setText("   ")

        # App UI Customization
        self.setWindowTitle("Signal Equalizer")
        self.setWindowIcon(QIcon("icon.png"))
        self.showMaximized()
        self.setStyleSheet('''
            QLabel {
                font-size: 14px;
                color: black;
            }
            QPushButton {
                background-color: white;
                color: black;
                border: 1px solid #CCCCFF;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #CCCCFF;
                color: black;
            }
            QSlider::handle:horizontal {
                background: #CCCCFF;
                border: 1px solid #CCCCFF;
                width: 20px;
            }
            QSlider::handle:vertical {
                background: #CCCCFF;
                border: 1px solid #CCCCFF;
                width: 20px;
            }
            QComboBox {
                background-color: white;
                border: 1px solid #CCCCFF;
                padding: 1px 18px 1px 3px;
            }
            QComboBox:hover {
                background-color: #CCCCFF;
                border: 1px solid #CCCCFF;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #CCCCFF;
            }
            QComboBox QAbstractItemView {
                background: #CCCCFF;
                border: 1px solid #CCCCFF;
            }
        ''')

        # Customize the style sheet for the btnChoose button
        self.btnChoose.setStyleSheet(
            '''
            QPushButton {
                background-color: #CCCCFF;
                color: black;
                border: 1px solid #CCCCFF;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: white;
                color: black;
            }
            '''
        )

        # Create PyQtGraph widget for the additional plot
        self.plotChoose = pg.PlotWidget()
        self.sliderChoose = QSlider(Qt.Horizontal)
        self.sliderChoose.setMinimum(1)
        self.sliderChoose.setMaximum(100)
        self.sliderChoose.setValue(0)
        self.btnClose = QPushButton()
        self.btnClose.setText("Ok")

        # slider zoom
        self.sliderZoom.setMinimum(-100)
        self.sliderZoom.setMaximum(100)
        self.sliderZoom.setValue(0)
        self.sliderZoom.setSingleStep(1)


        # slider speed
        self.sliderSpeed.setMinimum(1)
        self.sliderSpeed.setMaximum(99)
        self.sliderSpeed.setValue(1)
        self.sliderSpeed.setSingleStep(1)

        # slider pan vertical
        self.sliderPanV.setMinimum(-100)
        self.sliderPanV.setMaximum(100)
        self.sliderPanV.setValue(0)
        self.sliderPanV.setSingleStep(1)

        # slider pan horizontal
        self.sliderPanH.setMinimum(-100)
        self.sliderPanH.setMaximum(100)
        self.sliderPanH.setValue(0)
        self.sliderPanH.setSingleStep(1)

        # Set initial values, minimum, and maximum for sliders manually
        for i in range(1, 11):
            slider = getattr(self, f"slider{i}")
            slider.setValue(5)
            slider.setMinimum(0)
            slider.setMaximum(10)

        for i in range(1, 11):
            label = getattr(self, f"labelSliderValue_{i}")
            label.setText("")

        # Customize the plotChoose appearance
        self.plotChoose.setBackground('w')
        self.plotChoose.getAxis('bottom').setPen('k')
        self.plotChoose.getAxis('left').setPen('k')
        self.plotChoose.getAxis('bottom').setTextPen('k')
        self.plotChoose.getAxis('left').setTextPen('k')

        #Create a combo box
        self.boxWindow = QComboBox()
        window_types = ["Rectangle", "Hamming", "Hanning", "Gaussian"]
        self.boxWindow.addItems(window_types)

        # Create a QLabel 
        self.labelChoose = QLabel("Standard Deviation")

        # Add the PyQtGraph widget to the layout
        self.layoutChooseH = QHBoxLayout()

        # Create a new QFrame
        self.frameChoose = QFrame()

        # Set the layout of the new QFrame to QHBoxLayout
        self.frameChoose.setLayout(QHBoxLayout())
        self.labelChooseSlider = QLabel("0")

        # Add existing widgets to the new QFrame
        self.frameChoose.layout().addWidget(self.labelChoose)
        self.frameChoose.layout().addWidget(self.sliderChoose)
        self.frameChoose.layout().addWidget(self.labelChooseSlider)

        # Add the QFrame to the existing layout
        self.layoutChooseH.addWidget(self.frameChoose)
        self.layoutChooseH.addWidget(self.btnClose)
        self.frameChoose.hide()

        # Add the PyQtGraph widget to the layout
        self.layoutChoose = QVBoxLayout()
        self.layoutChoose.addWidget(self.boxWindow)
        self.layoutChoose.addWidget(self.plotChoose)
        self.layoutChoose.addLayout(self.layoutChooseH)

        # Create the pop-up dialog
        self.dialogChoose = QDialog(self)
        self.dialogChoose.setWindowTitle("Choose Window")
        self.dialogChoose.setLayout(self.layoutChoose)

        # Create PyQtGraph widgets for the three frames
        self.plotSignal = pg.PlotWidget()
        self.plotIn = pg.PlotWidget()
        self.plotOut = pg.PlotWidget()


        self.spectrogram_before = MatplotlibWidget()
        self.spectrograme_out = MatplotlibWidget()

        self.spectrograme_out.setMinimumSize(QtCore.QSize(300, 300))
        self.spectrograme_out.setMaximumSize(QtCore.QSize(600, 600))
        self.gridLayout.addWidget(self.spectrograme_out, 4, 2, 0, 0)

        self.spectrogram_before.fig.patch.set_facecolor((0.8,0.8,0.8))
        self.spectrograme_out.fig.patch.set_facecolor((0.8, 0.8, 0.8))


        self.spectrogram_before.setMinimumSize(QtCore.QSize(300, 300))
        self.spectrogram_before.setMaximumSize(QtCore.QSize(600, 600))
        self.gridLayout.addWidget(self.spectrogram_before, 4, 2, 0, 0)





        # Set layouts for the frames
        self.framePlot.setLayout(QVBoxLayout())
        self.frameSignalIn.setLayout(QVBoxLayout())
        self.frameSignalOut.setLayout(QVBoxLayout())
        self.frameSpecIn.setLayout(QVBoxLayout())
        self.frameSpecOut.setLayout(QVBoxLayout())

        # Customize the plot appearance
        for plot in [self.plotSignal, self.plotIn, self.plotOut]:
            plot.setBackground('w')
            plot.getAxis('bottom').setPen('k')
            plot.getAxis('left').setPen('k')
            plot.getAxis('bottom').setTextPen('k')
            plot.getAxis('left').setTextPen('k')

        # Add the PyQtGraph widgets to the frames
        self.framePlot.layout().addWidget(self.plotSignal)
        self.frameSignalIn.layout().addWidget(self.plotIn)
        self.frameSignalOut.layout().addWidget(self.plotOut)
        self.frameSpecIn.layout().addWidget(self.spectrogram_before)
        self.frameSpecOut.layout().addWidget(self.spectrograme_out)
        # Additional settings for the layout, if needed
        self.frameSpecIn.layout().setContentsMargins(0, 0, 0, 0)  # Adjust margins if necessary
        self.frameSpecIn.setLayout(self.frameSpecIn.layout())  # Set the layout for self.frameSpecIn

        # Update the background color for PyQtGraph widgets
        self.plotChoose.setBackground('#333')
        for plot in [self.plotSignal, self.plotIn, self.plotOut]:
            plot.setBackground('#333')

        # Update the color for PyQtGraph axes
        for plot in [self.plotChoose, self.plotSignal, self.plotIn, self.plotOut]:
            plot.getAxis('bottom').setPen(QColor(200, 200, 200))  # Light gray
            plot.getAxis('left').setPen(QColor(200, 200, 200))
            plot.getAxis('bottom').setTextPen(QColor(200, 200, 200))
            plot.getAxis('left').setTextPen(QColor(200, 200, 200))


        # Triggers
        self.logic_app = Logic(self)
        self.btnOpen.clicked.connect(lambda:self.logic_app.load_signal())
        self.btnPlayin.clicked.connect(lambda:self.logic_app.play_pause())
        self.btnHide.clicked.connect(lambda:self.logic_app.toggle_visibility())
        self.slider1.valueChanged.connect(lambda :self.logic_app.slider_1())
        self.slider2.valueChanged.connect(lambda: self.logic_app.slider_2())
        self.slider3.valueChanged.connect(lambda: self.logic_app.slider_3())
        self.slider4.valueChanged.connect(lambda: self.logic_app.slider_4())
        self.slider5.valueChanged.connect(lambda: self.logic_app.slider_selection())
        self.slider6.valueChanged.connect(lambda: self.logic_app.slider_selection())
        self.slider7.valueChanged.connect(lambda: self.logic_app.slider_selection())
        self.slider8.valueChanged.connect(lambda: self.logic_app.slider_selection())
        self.slider9.valueChanged.connect(lambda: self.logic_app.slider_selection())
        self.slider10.valueChanged.connect(lambda: self.logic_app.slider_selection())
        self.btnPlayout.clicked.connect(lambda :self.logic_app.play_modified_signal())
        self.boxWindow.currentIndexChanged.connect(lambda index: self.logic_app.change_window(index))
        self.boxMode.currentIndexChanged.connect(lambda index: self.logic_app.handle_mode_change(index))
        self.btnClose.clicked.connect(self.closeDialog)
        self.btnChoose.clicked.connect(self.showChooseDialog)
        self.sliderChoose.valueChanged.connect(lambda: self.logic_app.change_window(self.boxWindow.currentIndex()))
        self.btnReset.clicked.connect(self.logic_app.reset_all)
        self.btnStop.clicked.connect(self.logic_app.stop_and_reset)
        self.sliderZoom.valueChanged.connect(self.logic_app.update_plots)
        self.sliderSpeed.valueChanged.connect(self.logic_app.update_speed)
        self.sliderPanH.valueChanged.connect(self.logic_app.update_pan_horizontal)
        self.sliderPanV.valueChanged.connect(self.logic_app.update_pan_vertical)


        # slider connections
        self.slider1.valueChanged.connect(lambda value: self.labelSliderValue_1.setText(str(value)))
        self.slider2.valueChanged.connect(lambda value: self.labelSliderValue_2.setText(str(value)))
        self.slider3.valueChanged.connect(lambda value: self.labelSliderValue_3.setText(str(value)))
        self.slider4.valueChanged.connect(lambda value: self.labelSliderValue_4.setText(str(value)))
        self.slider5.valueChanged.connect(lambda value: self.labelSliderValue_5.setText(str(value)))
        self.slider6.valueChanged.connect(lambda value: self.labelSliderValue_6.setText(str(value)))
        self.slider7.valueChanged.connect(lambda value: self.labelSliderValue_7.setText(str(value)))
        self.slider8.valueChanged.connect(lambda value: self.labelSliderValue_8.setText(str(value)))
        self.slider9.valueChanged.connect(lambda value: self.labelSliderValue_9.setText(str(value)))
        self.slider10.valueChanged.connect(lambda value: self.labelSliderValue_10.setText(str(value)))
        self.sliderChoose.valueChanged.connect(lambda value: self.labelChooseSlider.setText(str(value)))


    def actual_value(self, new_value,name):

            if name=="slider_1":
                self.labelSliderValue_1.setText(f"{new_value} ")
            elif name=="slider_2" :
                self.labelSliderValue_2.setText(f"{new_value} ")
            elif name=="slider_3" :
                self.labelSliderValue_3.setText(f"{new_value} ")
            elif name=="slider_4" :
                self.labelSliderValue_4.setText(f"{new_value} ")


    def closeDialog(self):
        # Close the dialog when called
        self.dialogChoose.close()

    def showChooseDialog(self):
        # Show the pop-up dialog
        self.dialogChoose.exec_()



def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()
