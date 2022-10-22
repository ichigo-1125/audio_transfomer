################################################################################
# 音声データセットの読み込み
################################################################################
from data_reader import DataReaderBase
import numpy as np
import librosa
import wave
import glob

# サンプリングレート
SAMPLING_RATE = 441

# ファイル種別定数
DATA_TYPE_WAV = 501
DATA_TYPE_MP3 = 502

class AudioDataReader(DataReaderBase):

    ############################################################################
    # データの一括読み込み
    ############################################################################
    def read_dataset( self ):

        if self.data_type == DATA_TYPE_WAV:

            # ディレクトリ配下のファイルリストを取得して読み込む
            file_list = glob.glob(self.data_dir + "*.wav")
            for file in file_list:
                self.datas.append(self.read_from_wav_file(file))

            self.is_ready = True

        else:
            self.datas = []
            self.is_ready = False

    ############################################################################
    # データ種別が有効かどうか
    ############################################################################
    def check_data_type( self, data_type ):
        if data_type == DATA_TYPE_WAV:
            return True
        else:
            return False

    ############################################################################
    # wavファイルからデータを読み込む
    ############################################################################
    def read_from_wav_file( self, file_path ):
        data, rate = librosa.load(file_path, sr=SAMPLING_RATE)
        return { 'rate' : rate, 'data' : data }
