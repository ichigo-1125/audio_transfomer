from audio_data_reader import AudioDataReader, DATA_TYPE_WAV
import matplotlib.pyplot as plt
import numpy as np
import math

INPUT_SIZE = 441

def softmax(x):
    u = np.sum(np.exp(x))
    return np.exp(x) / u

# データの読み込み
inst = AudioDataReader(data_dir = "./data", data_type = DATA_TYPE_WAV)
dataset = inst.get_dataset()

data = dataset[0]['data'] / (2**16 / 2)
fft_data = np.abs(np.fft.fft(data))
rate = dataset[0]['rate']

# FFT終わったあと
freq_list = np.fft.fftfreq(data.shape[0], d=1.0/rate)

# ラベル
labels = ["positive", "negative"]
label_num = len(labels)

# input
frame_num = math.ceil(data.size / rate)
frames = np.resize(data, [frame_num, INPUT_SIZE])

result1 = np.dot(frames, np.random.rand(INPUT_SIZE, 1))
result2 = np.dot(result1.T, np.random.rand(frame_num, label_num))
probability = softmax(result2)
print(probability)
print(labels[probability.argmax()])
