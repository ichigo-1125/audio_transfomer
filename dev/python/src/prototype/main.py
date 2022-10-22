from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pprint
import torch

# トークン化
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")

# モデル
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

# 分類したいクラス
classes = ["not paraphrase", "is paraphrase"]

# サンプルテキスト
sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

# サンプルテキストをベクトル化したもの
tokenized_text = tokenizer.tokenize(sequence_0)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])
paraphrase = tokenizer.encode(sequence_0, return_tensors="pt")

print(tokenized_text)
print(indexed_tokens)
print(tokens_tensor)
print(paraphrase)
