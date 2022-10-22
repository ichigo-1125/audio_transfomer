import random
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer

################################################################################
# 前処理
################################################################################
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
    )
    return inputs

# 学習用データセット
minds = load_dataset("PolyAI/minds14", name="en-US", split="train")
minds = minds.train_test_split(test_size=0.2)
print("=== Load Datasets ===")
print(minds)
print("=====================\n\n")

# 不要なカラムを削除
minds = minds.remove_columns(["path", "transcription", "english_transcription", "lang_id"])
print("=== Adjast Datasets ===")
print(minds)
print(minds["train"][0]["audio"]["array"])
print(len(minds["train"][0]["audio"]["array"]))
print("=====================\n\n")

# 分類ラベル
# labels = minds["train"].features["intent_class"].names
labels = ['positive', 'negative']
print("=== Labels ===")
print(labels)

# ラベルとIDの変換
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
print(label2id)
print(id2label)
print("=====================\n\n")
 
# 特徴抽出
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# サンプリング
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
print("=== Sampling ===")
print(minds)
print(minds["train"][0]["audio"]["array"])
print(len(minds["train"][0]["audio"]["array"]))
print("=====================\n\n")

# サンプリングデータに対して前処理を実行
encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)
encoded_minds = encoded_minds.rename_column("intent_class", "label")
print("=== Preprocess ===")
print(encoded_minds)
print("=====================\n\n")

# 学習
num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    num_train_epochs=5,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_minds["train"],
    eval_dataset=encoded_minds["test"],
    tokenizer=feature_extractor,
)
trainer.train()
