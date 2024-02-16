from PyQt5.QtWidgets import QFileDialog
import numpy as np
import os
import pyqtgraph as pg
from scipy.io import wavfile
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5 import QtCore
from PyQt5.QtCore import QUrl, QTimer
import vlc
from pydub import AudioSegment,effects

from pydub.playback import play
from scipy.signal import gaussian
import pyaudio
import pygame
import array

def windows( name, length):
    if name == "gaussian":
        std_dev = 10
        # Generate the Gaussian window
        window_signal = gaussian(length, std_dev)

    elif name == "Rectangle":
        window_signal = np.ones(length)

    elif name == 'hamming':
        window_signal = np.hamming(length)

    elif name == 'hanning':
        window_signal = np.hanning(length)
    return window_signal



if __name__ == "__main__":
    list=windows("Rectangle",2501)
    print(list)