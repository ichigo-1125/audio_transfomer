# ゴール

入力: 音声データ
出力: ポジティブ、ネガティブ


# 時系列データ

2001年 1000円
2002年 2000円

私はパンを食べた。

私　は　パン　を　食べた

私 [1, 2, 3]
は [2, 3, 1]
パン [3, 1, 2]
ご飯 [3, 2, 2]

xxxで嬉しい
xxxで悲しい

音 [高さ, 大きさ]


# Transformer

エンコーダを作ればオッケー
入力↑に書いたベクトル
出力は数値-1〜1


# Attention

Transformerのコア的なレイヤ
RNNだと並列化できなかった計算を並列化してGPUリソースを最大限活用できるようにした。


# 音声データの前処理

- 1次元の周波数のプロットの集まり
data : プロットの集まり
rate : フレームレート

```
# フレーム数
len = data.length() / rate

preprocessed_data = []
for i in range(0, len):
	min = i*rate
	max = (i+1)*rate-1
	frame = data[min:max]

	fft_frame =
	spct_frame =
	preprocessed_data.append(spect_frame)

[
	[
		spect_frame,
		spect_frame,
		spect_frame,
		spect_frame,
	],
	[
	]
]

# 内積をとって入力サイズを合わせる
[100, 1] . [50, 100] = [50, 1]
[30, 1] . [50, 30] = [50, 1]
```


//	入力データ
[
	a11, a12, a13, a14, ...., a1512,
	a21, a22, a23, a24, ...., a2512,
	a31, a32, a33, a34, ...., a3512,
	a41, a42, a43, a44, ...., a4512,
]

.

[
	b1,
	b2,
	b3,
	...
	b512,
]

=

[
	b1,
	b2,
	b3,
	...
	b512,
]

[0.7, 0.42] => softmax => [0.65, 0.35]
