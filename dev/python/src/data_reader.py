################################################################################
# データセットの読み込み
################################################################################

# ファイル種別定数
DATA_TYPE_UNSELECTED = 0

class DataReaderBase:

    # データの準備ができているか
    is_ready = False

    # データの実体
    datas = []

    # データを読み込むパス
    data_dir = ""

    # 読み込むデータの種別
    data_type = DATA_TYPE_UNSELECTED

    ############################################################################
    # コンストラクタ
    ############################################################################
    def __init__( self, data_dir = "", data_type = DATA_TYPE_UNSELECTED ):
        self.set_data_dir(data_dir)
        self.set_data_type(data_type)

    ############################################################################
    # データ読み込みパスの設定
    ############################################################################
    def set_data_dir( self, data_dir ):

        # スラッシュがなければ自動で追加
        if len(data_dir) > 0 and data_dir.endswith("/") == False:
            data_dir += "/"

        self.data_dir = data_dir

    ############################################################################
    # データ種別の設定
    ############################################################################
    def set_data_type( self, data_type ):
        if self.check_data_type(data_type) == False:
            print(data_type + " is unsupported data format")
            data_type = DATA_TYPE_UNSELECTED

        # 内部データの種別が変わった場合
        if self.data_type != data_type:
            self.is_ready = False
            self.data = []
            self.data_type = data_type

    ############################################################################
    # データセットを取得
    ############################################################################
    def get_dataset( self ):
        if self.is_ready == False:
            self.read_dataset()

        return self.datas

    ############################################################################
    # データの一括読み込み
    ############################################################################
    def read_dataset( self ):
        print("`DataReader.read_dataset()` is not implemented")

    ############################################################################
    # データ種別が有効かどうか
    ############################################################################
    def check_data_type( self, data_type ):
        print("`DataReader.check_data_type()` is not implemented")
        return False;
