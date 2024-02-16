from PyQt5.QtWidgets import QFileDialog
import threading
from scipy.io import wavfile
from PyQt5 import QtCore
import sys, os, shutil
from librosa.core.spectrum import _spectrogram
import numpy as np

import librosa.display

from scipy.signal import gaussian
import pyaudio

import vlc
import pandas as pd
from pydub import AudioSegment

AudioSegment.converter = "C:\\Program Files (x86)\\ffmpeg-master-latest-win64-gpl\\ffmpeg.exe"


class Logic():
    def __init__(self, ui_instance):
        self.ui_instance = ui_instance
        self.frequency_range = np.linspace(0, 1, 1000)
        self.spectrogram_widget = {

            'before': self.ui_instance.spectrogram_before,
            'out': self.ui_instance.spectrograme_out
        }
        self.spectrogram_time_min, self.spectrogram_time_max = 0, 0
        self.spectrogram_widget['before'].toolbar.hide()
        self.spectrogram_widget['out'].toolbar.hide()
        self.start1 = None
        self.end1 = None
        self.timer = QtCore.QTimer()
        self.timer.setInterval(200)
        self.std = 0
        self.xMin = 0
        self.xMax = 10
        self.f_rate = 0
        self.SIZE = 0

        self.spectrograms_visible = True
        self.freq_start = None
        self.freq_end = None
        self.start2 = None
        self.end2 = None
        self.selected_mode = None
        self.window_signal = None
        self.original_slider_values = {
            'slider1': self.ui_instance.slider1.value(),
            'slider2': self.ui_instance.slider2.value(),
            'slider3': self.ui_instance.slider3.value(),
            'slider4': self.ui_instance.slider4.value(),
            'slider5': self.ui_instance.slider5.value(),
            'slider6': self.ui_instance.slider6.value(),
            'slider7': self.ui_instance.slider7.value(),
            'slider8': self.ui_instance.slider8.value(),
            'slider9': self.ui_instance.slider9.value(),
            'slider10': self.ui_instance.slider10.value()}
        self.start3 = None
        self.end3 = None

    def load_signal(self):
        self.fname = QFileDialog.getOpenFileName(
            None, "Select a file...", os.getenv('HOME'), filter="All files (*)"
        )
        if not self.fname[0]:
            print("No file selected")
            return

        path = self.fname[0]
        if path.lower().endswith('.csv'):
            self.ECG_signals(path)
        else:
            self.sound_file(path)

    def common_setup(self):
        self.plot(self.ui_instance.plotIn, self.xAxisData, self.yData, "r")
        self.convert_frequancy(self.yData,self.f_rate)
        self.flage = 1

        self.timer.start()
        self.update_slider_labels()

    def sound_file(self, path):
        if '.mp3' in path:
            song = AudioSegment.from_mp3(path)
            song.export(r"./final.wav", format="wav")
            self.f_rate, self.yData = wavfile.read(r"./final.wav")
        else:
            self.f_rate, self.yData = wavfile.read(path)

        if len(self.yData.shape) > 1:
            self.yData = self.yData[:, 0]
        self.SIZE = len(self.yData)

        self.yData = self.yData / (2.0 ** 15) * 4
        self.yAxisData = self.yData
        self.xAxisData = np.linspace(0, self.SIZE / self.f_rate, num=self.SIZE)

        self.common_setup()
        self.plot_spectrogram(self.yData, self.f_rate, 'before')
        # self.init_plot_ranges()

        self.player = vlc.MediaPlayer(path)
        amplification_factor = 2.0  # You can adjust this factor as needed
        new_volume = min(100, int(50 * amplification_factor))  # Ensure it's within the valid range
        self.player.audio_set_volume(new_volume)

        self.player.play()
        self.ui_instance.btnPlayin.setText("Pause")

    def ECG_signals(self, path):
        try:
            filename = os.path.basename(path)
            self.name, _ = os.path.splitext(filename)
            print(
                self.name
            )
            df = pd.read_csv(path)
            columns = df.columns
            time = df[columns[0]].values
            signal = df[columns[1]].values

            self.f_rate = 500
            self.yData = signal.flatten()
            self.yAxisData = self.yData

            self.xAxisData = time
            self.SIZE = len(self.yData)
            self.common_setup()
            self.ui_instance.specXmin = 0.6
            self.ui_instance.specXmax = 1.6
            self.plotSignalxleft = -150
            self.plotSignalxright = 150
            self.plot_spectrogram(self.yData, self.f_rate, 'before')
            self.plot(self.ui_instance.plotOut, time, signal, 'r')
            self.convert_frequancy(self.yData, self.f_rate)

        except KeyError as e:
            print(f"Error: {e} column not found in the CSV file.")

    def convert_frequancy(self, signal, sampling_rate):

        N = len(signal)
        T = 1.0 / sampling_rate
        yf = 2 * np.fft.rfft(signal)
        self.fft_result = yf
        yf[0] = 0
        frequencies = np.fft.rfftfreq(N, T)
        self.freq = frequencies
        wedget = self.ui_instance.plotSignal

        if len(frequencies) != len(yf):
            frequencies = frequencies[:len(yf)]
            # Plot the data

        self.plot(wedget, frequencies, np.abs(yf), 'b')


    def inv_fourier_transform(self, frequencies, yf, sampling_rate):
        N = len(frequencies) * 2 - 2  # Determine the length of the original signal
        T = 1.0 / sampling_rate
        signal_reconstructed = np.fft.irfft(yf)
        time_values = np.linspace(0.0, N * T, N, endpoint=False)
        wedget = self.ui_instance.plotOut
        self.plot(wedget, time_values, signal_reconstructed, 'r')
        self.plot_spectrogram(signal_reconstructed, self.f_rate, 'out')
        return time_values, signal_reconstructed

    def plot(self, wedget, x_data, y_data, color):
        wedget.clear()
        wedget.plot(
            x_data, y_data, linewidth=100, pen=color)

   

   

    def windows(self, name, length):
        window_functions = {
            "Gaussian": lambda length: gaussian(length, self.ui_instance.sliderChoose.value()),
            "Rectangle": lambda length: np.ones(length),
            "hamming": np.hamming,
            "Hanning": np.hanning,
        }

        if name in window_functions:
            self.window_signal = window_functions[name](length)

        else:
            raise ValueError(f"Unsupported window name: {name}")

        return self.window_signal

    def set_slider_values(self, slider_number):
        mode_values = {
            "Musical Instruments Mode": [
                (0, 1000, 0, -1000),
                (1000, 2000, -1000, -2000),
                (7000, 15000, -7000, -15000),
                (0, 3500, 0, -3500)
            ],
            "Animal Sounds Mode": [
                (0, 900, 0, -900),
                (1001, 1800, -1001, -1800),
                (1801, 4000, -1801, -4000),
                (4001, 24000, -4001, -24000)
            ],
            "Uniform Range Mode": [
                (0, 1000),
                (1001, 2000),
                (2001, 3000),
                (4001, 5000)
            ],
            "ECG Abnormalities Mode": [
                (0,5),
                (5, 10),
                (10,35)
            ],
            # Add more modes and their corresponding values as needed
        }

        if self.selected_mode in mode_values and slider_number <= len(mode_values[self.selected_mode]):
            values = mode_values[self.selected_mode][slider_number - 1]
            self.freq_start, self.freq_end, *rest = values

            if self.selected_mode.startswith("Musical Instruments Mode") or self.selected_mode.startswith(
                    "Animal Sounds Mode"):
                self.play_pause()

            # Additional mode-specific logic if needed

    def update_slider(self, slider_number):
        value = int(getattr(self.ui_instance, f"slider{slider_number}").value())
        self.ui_instance.actual_value(value, f"slider_{slider_number}")
        self.splite_chosing_freq(value)

    def slider_1(self):
        self.set_slider_values(1)
        self.update_slider(1)

    def slider_2(self):
        self.set_slider_values(2)
        self.update_slider(2)

    def slider_3(self):
        self.set_slider_values(3)
        self.update_slider(3)

    def slider_4(self):
        self.set_slider_values(4)
        self.update_slider(4)

    def splite_chosing_freq(self, value):
        if self.flage:
            self.modfied_data = np.copy(self.fft_result)
            self.flage = 0
        freq_mask = (self.freq >= self.freq_start) & (self.freq <= self.freq_end)
        filtered_freq = self.freq[freq_mask]
        filtered_magnitude = np.abs(self.fft_result[freq_mask])

        data_magnitude = np.array(filtered_magnitude)

        # Generate window
        selected_window = self.ui_instance.boxWindow.currentText()

        data_window = self.windows(selected_window, len(data_magnitude))

        data_window_gained = np.array(data_window * value / 5)
        final_data = data_magnitude * data_window_gained

        self.modfied_data[freq_mask] = final_data
        if value==5:
            self.modfied_data=np.copy(self.fft_result)
            print(value)

        self.plot(self.ui_instance.plotSignal, self.freq, np.abs(self.modfied_data), 'b')
        # self.inverse_fourier_transform(self.freq,self.modfied_data, self.fft_phase)
        self.inv_fourier_transform(self.freq, self.modfied_data, self.f_rate)

    def plot_spectrogram(self, samples, sampling_rate, widget):
        self.spectrogram_widget[widget].getFigure().clear()

        # Access the toolbar and hide it
        spectrogram_axes = self.spectrogram_widget[widget].getFigure().add_subplot(111)

        # Ensure that 'data' is a one-dimensional array
        data = samples.astype('float32').ravel()

        # Choose a suitable value for n_fft based on the length of the input signal
        n_fft = min(2048, len(data))

        # Compute the spectrogram
        frequency_magnitude = np.abs(librosa.stft(data, n_fft=n_fft)) ** 2
        mel_spectrogram = librosa.feature.melspectrogram(S=frequency_magnitude, sr=sampling_rate, n_mels=1024)

        # Convert power spectrogram to dB scale
        decibel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Display the spectrogram
        spectrogram_image = librosa.display.specshow(decibel_spectrogram, x_axis='time', y_axis='mel', sr=sampling_rate,
                                                     ax=spectrogram_axes)

        colorbar = self.spectrogram_widget[widget].getFigure().colorbar(spectrogram_image, ax=spectrogram_axes,
                                                                        format='%+2.0f dB')

        # Get the x-axis limits from the colorbar
        x_range = colorbar.ax.get_xlim()
        self.spectrogram_time_min, self.spectrogram_time_max = x_range

        # Draw the spectrogram
        self.spectrogram_widget[widget].draw()

    def inverse_fourier_transform(self, freq, magnitude, phase):
        # Close all open figures
        # plt.close('all')

        complex_signal = magnitude * np.exp(1j * phase)

        # Apply the Inverse Fourier Transform
        self.magnitude_in_time = np.fft.ifft(complex_signal)

        # Ensure that the magnitude in time is real-valued
        self.magnitude_in_time = np.real(self.magnitude_in_time)

        self.plot(self.ui_instance.plotOut, self.xAxisData, self.magnitude_in_time, 'b')

        self.plot_spectrogram(self.magnitude_in_time, self.f_rate, 'out')

    def play_modified_signal(self):
        # Pause the player if it's playing
        self.play_pause()

        # Use a separate thread for audio playback
        playback_thread = threading.Thread(target=self.play_audio_thread,
                                           args=(self.freq, self.modfied_data, self.fft_phase))
        playback_thread.start()

    def play_audio_thread(self, freq, magnitude, phase):
        audio_data = self.synthesize_audio(freq, magnitude, phase)
        self.play_audio(audio_data)

    def synthesize_audio(self, freq, magnitude, phase):
        # Synthesize the audio signal using the inverse Fourier transform
        complex_signal = magnitude * np.exp(1j * phase)
        audio_signal = np.fft.ifft(complex_signal).real

        # Normalize the audio signal to the range [-1, 1]
        audio_signal /= np.max(np.abs(audio_signal))

        return audio_signal

    def play_audio(self, audio_data):
        # Create a PyAudio stream
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=self.f_rate,
                        output=True)

        # Play the audio in chunks
        chunk_size = 1024
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size].astype(np.float32).tobytes()
            stream.write(chunk)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

    def play_pause(self):
        if self.player.is_playing():
            self.player.pause()
            self.timer.stop()
            self.ui_instance.btnPlayin.setText("Playin")
        else:
            self.player.play()
            self.timer.start()
            self.ui_instance.btnPlayin.setText("Pause")

    def change_window(self, index):
        window_types = {
            1: 'rectangle',
            2: 'hamming',
            3: 'hanning',
            4: 'gaussian'
        }

        # Default window type
        window_type = window_types.get(index, 'rectangle')

        # Show or hide the frame based on the selected window type
        self.ui_instance.frameChoose.setVisible(window_type == 'gaussian')

        # Call the create_window function
        self.create_window(window_type)

    def create_window(self, window_type):
        self.ui_instance.plotChoose.clear()
        self.std = self.ui_instance.sliderChoose.value() / 100
        # Calculate the window signal based on the selected window_type
        if window_type == 'rectangle':
            window_signal = np.ones_like(self.frequency_range)
        elif window_type == 'hamming':
            window_signal = np.hamming(len(self.frequency_range))
        elif window_type == 'hanning':
            window_signal = np.hanning(len(self.frequency_range))
        elif window_type == 'gaussian':
            window_signal = np.exp(-(0.5 * ((self.frequency_range - 0.5) / self.std)) ** 2)

        self.plot_curve = self.ui_instance.plotChoose.plot(self.frequency_range, window_signal, pen='b')

    def handle_mode_change(self, index):
        # Get the selected mode from the combo box
        self.selected_mode = self.ui_instance.boxMode.currentText()

        # Define a mapping between modes and labels
        mode_labels = {
            "Uniform Range Mode": ["0-1000", "1000-2000", "2000-3000", "3000-4000", "4000-5000", "5000-6000",
                                   "6000-7000", "7000-8000", "8000-9000", "9000-1000"],
            "Musical Instruments Mode": ["Guitar", "piccolo", "xylphone", "drum"],
            "Animal Sounds Mode": ["dog", "cow", "cat", "bird"],
            "ECG Abnormalities Mode": ["Signal 1", "Signal 2", "Signal 3", "Signal 4"]
        }

        # Handle the visibility of sliders and frames based on the selected mode
        self.ui_instance.frame6Sliders.show() if self.selected_mode == "Uniform Range Mode" else self.ui_instance.frame6Sliders.hide()
        self.ui_instance.frame.show()
        if self.selected_mode == "ECG Abnormalities Mode":
            self.ui_instance.slider4.hide()
            self.ui_instance.labelSlider4.hide()
        else:
            self.ui_instance.slider4.show()
            self.ui_instance.labelSlider4.show()

        # Update slider labels based on the selected mode
        labels = mode_labels.get(self.selected_mode, [])
        for i, label in enumerate(labels, start=1):
            getattr(self.ui_instance, f"labelSlider{i}").setText(label)

    def update_slider_labels(self):
        if self.selected_mode == "Uniform Range Mode":
            for i in range(1, 11):
                slider_label = getattr(self.ui_instance, f'labelSlider{i}')

                if self.f_rate == 0:
                    slider_label.setText("")
                else:
                    slider_label.setText(f"{i * 1000} Hz")

    def toggle_visibility(self):
        if self.spectrograms_visible:
            # Hide the spectrograms and update the button text
            self.ui_instance.frameSpecIn.hide()
            self.ui_instance.frameSpecOut.hide()
            self.ui_instance.inputSpec.hide()
            self.ui_instance.outSpec.hide()
            self.ui_instance.btnHide.setText("Unhide Spectrograms")
        else:
            # Show the spectrograms and update the button text
            self.ui_instance.frameSpecIn.show()
            self.ui_instance.frameSpecOut.show()
            self.ui_instance.inputSpec.show()
            self.ui_instance.outSpec.show()
            self.ui_instance.btnHide.setText("Hide Spectrograms")

        # Toggle the visibility state
        self.spectrograms_visible = not self.spectrograms_visible

    def reset_all(self):
        # Stop the timer if running
        self.timer.stop()

        # Reset sliders to their original values
        self.ui_instance.slider1.setValue(self.original_slider_values['slider1'])
        self.ui_instance.slider2.setValue(self.original_slider_values['slider2'])
        self.ui_instance.slider3.setValue(self.original_slider_values['slider3'])
        self.ui_instance.slider4.setValue(self.original_slider_values['slider4'])
        self.ui_instance.sliderZoom.setValue(0)
        self.ui_instance.sliderSpeed.setValue(0)
        self.ui_instance.sliderPanH.setValue(0)
        self.ui_instance.sliderPanV.setValue(0)
        # Reset other sliders as needed

        # Clear all plots
        self.ui_instance.plotIn.clear()
        self.ui_instance.plotSignal.clear()
        self.ui_instance.plotOut.clear()
        self.ui_instance.spectrogramIn.clear()
        self.ui_instance.spectrogramOut.clear()

        # Reset other state variables as needed
        self.freq_start = None
        self.freq_end = None
        self.selected_mode = None

        # Restart the timer if needed
        self.timer.start()

    def stop_and_reset(self):
        # Pause the player if it is playing
        if self.player.is_playing():
            self.player.pause()
            self.timer.stop()
            self.playing = False
            self.ui_instance.btnPlayin.setText("Playin")

        # Reset the player to the beginning
        self.player.set_media(self.player.get_media())
        self.ui_instance.plotOut.clear()

    def update_plots(self, value):
        zoom_factor = self.ui_instance.sliderZoom.value() / 100

        # update zoom
        self.ui_instance.plotIn.setXRange(
            self.ui_instance.xmin_plot_in + self.ui_instance.middle_offset_plot_in * zoom_factor,
            self.ui_instance.xmax_plot_in - self.ui_instance.middle_offset_plot_in * zoom_factor)
        self.ui_instance.plotOut.setXRange(
            self.ui_instance.xmin_plot_out + self.ui_instance.middle_offset_plot_out * zoom_factor,
            self.ui_instance.xmax_plot_out - self.ui_instance.middle_offset_plot_out * zoom_factor)

        self.init_after_zoom()

    def update_speed(self, value):
        speed_factor = 1 + self.ui_instance.sliderSpeed.value() / 100

        # Adjust the speed of the player
        self.player.set_rate(speed_factor)

    def init_after_zoom(self):
        # current ends of the xrange
        self.ui_instance.curr_xmin_plot_in = self.ui_instance.plotIn.viewRange()[0][0]
        self.ui_instance.curr_xmax_plot_in = self.ui_instance.plotIn.viewRange()[0][1]
        self.ui_instance.curr_xmin_plot_out = self.ui_instance.plotOut.viewRange()[0][0]
        self.ui_instance.curr_xmax_plot_out = self.ui_instance.plotOut.viewRange()[0][1]

        self.ui_instance.curr_xmin_plot_signal = self.ui_instance.plotSignal.viewRange()[0][0]
        self.ui_instance.curr_xmax_plot_signal = self.ui_instance.plotSignal.viewRange()[0][1]

        # current middle_offset
        self.ui_instance.curr_middle_offset_plot_in = (self.ui_instance.plotIn.viewRange()[0][1] +
                                                       self.ui_instance.plotIn.viewRange()[0][0]) / 2

        self.ui_instance.curr_middle_offset_plot_signal = (self.ui_instance.plotSignal.viewRange()[0][1] +
                                                           self.ui_instance.plotSignal.viewRange()[0][0]) / 2

        # current ends of the xrange
        self.ui_instance.curr_xmin_plot_in = self.ui_instance.plotIn.viewRange()[0][0]
        self.ui_instance.curr_xmax_plot_in = self.ui_instance.plotIn.viewRange()[0][1]
        self.ui_instance.curr_xmin_plot_out = self.ui_instance.plotOut.viewRange()[0][0]
        self.ui_instance.curr_xmax_plot_out = self.ui_instance.plotOut.viewRange()[0][1]

        self.ui_instance.curr_xmin_plot_signal = self.ui_instance.plotSignal.viewRange()[0][0]
        self.ui_instance.curr_xmax_plot_signal = self.ui_instance.plotSignal.viewRange()[0][1]

    def update_pan_horizontal(self, value):
        panh_factor = self.ui_instance.sliderPanH.value() / 100

        # update pan horizontal
        self.ui_instance.plotIn.setXRange(
            self.ui_instance.curr_xmin_plot_in + self.ui_instance.curr_middle_offset_plot_in * panh_factor,
            self.ui_instance.curr_xmax_plot_in + self.ui_instance.curr_middle_offset_plot_in * panh_factor)
        self.ui_instance.plotOut.setXRange(
            self.ui_instance.curr_xmin_plot_out + self.ui_instance.curr_middle_offset_plot_out * panh_factor,
            self.ui_instance.curr_xmax_plot_out + self.ui_instance.curr_middle_offset_plot_out * panh_factor)

        self.ui_instance.spectrogramOut.setXRange(
            self.ui_instance.curr_xmin_spectrogram_out + self.ui_instance.curr_middle_offset_spectrogram_out * panh_factor,
            self.ui_instance.curr_xmax_spectrogram_out + self.ui_instance.curr_middle_offset_spectrogram_out * panh_factor)

    def update_pan_vertical(self, value):
        panv_factor = self.ui_instance.sliderPanV.value() / 100

        # update pan vertical
        self.ui_instance.plotIn.setYRange(
            self.ui_instance.ymin_plot_in - self.ui_instance.middle_offset_plot_in_y * panv_factor * 10,
            self.ui_instance.ymax_plot_in - self.ui_instance.middle_offset_plot_in_y * panv_factor * 10)
        self.ui_instance.plotOut.setYRange(
            self.ui_instance.ymin_plot_out + self.ui_instance.middle_offset_plot_out_y * panv_factor,
            self.ui_instance.ymax_plot_out + self.ui_instance.middle_offset_plot_out_y * panv_factor)
