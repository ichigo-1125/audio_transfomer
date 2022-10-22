from audio_data_reader import AudioDataReader, DATA_TYPE_WAV
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = 441
BATCH_SIZE = 1

class Net( nn.Module ):

    # 使用するオブジェクトを定義
    def __init__( self, input_size ):
        super(Net, self).__init__()
        self.fc1 = nn.Flatten()
        self.fc2 = nn.Linear(input_size, 100)
        self.fc3 = nn.Linear(100, 2)

    # 順伝播
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def softmax(x):
    u = np.sum(np.exp(x))
    return np.exp(x) / u

# データの読み込み
inst = AudioDataReader(data_dir = "./data", data_type = DATA_TYPE_WAV)
dataset = inst.get_dataset()

# ラベル
label_symbols = ["positive", "negative"]
labels = [1, 2]
label_num = len(labels)

# 前処理
p_data = []
p_label = []
for raw_data in dataset:
    # データ
    data = raw_data['data'] / (2**16 / 2)
    rate = raw_data['rate']

    # FFT終わったあと
    fft_data = np.abs(np.fft.fft(data))
    freq_list = np.fft.fftfreq(data.shape[0], d=1.0/rate)

    # input
    frame_num = math.ceil(data.size / rate)
    data.resize(frame_num, INPUT_SIZE)

    # 前処理済みデータを配列に入れる
    p_data.append(data)
    p_label.append(random.randint(1, 3))

# 最大のフレーム数を算出
max_frame = 0
for data in p_data:
    if max_frame < len(data):
        max_frame = len(data)

# 最大のフレーム数のデータに合わせてリサイズ
for data in p_data:
    data.resize(max_frame, INPUT_SIZE, refcheck=False)

# torch用にデータを変形
torch.manual_seed(0)
t_data = torch.tensor(np.array(p_data), dtype=torch.float32)
t_label = torch.tensor(p_label, dtype=torch.int64)
t_dataset = torch.utils.data.TensorDataset(t_data, t_label)

# データセット分割
n_train = int(len(t_dataset) * 0.6)
n_val = int(len(t_dataset) * 0.2)
n_test = len(t_dataset) - n_train - n_val

train, val, test = torch.utils.data.random_split(t_dataset, [n_train, n_val, n_test])

train_loader = torch.utils.data.DataLoader(train, BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test, BATCH_SIZE)

# モデルの初期化
torch.manual_seed(0)

# モデルのインスタンス化とデバイスへの転送
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = Net(INPUT_SIZE * max_frame).to(device)

# 目的関数の設定
criterion = F.cross_entropy

# オプティマイザ
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# エポックの数
max_epoch = 1

for epoch in range(max_epoch):

    for batch in train_loader:
        print(batch)

        x, t = batch
        x = x.to(device)
        t = t.to(device)
        optimizer.zero_grad()
        y = net(x)
        loss = criterion(y, t)

        # New：正解率の算出
        y_label = torch.argmax(y, dim=1)
        acc  = torch.sum(y_label == t) * 1.0 / len(t)
        print('accuracy:', acc)

        loss.backward()
        optimizer.step()
