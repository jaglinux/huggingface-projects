from transformers import BertModel, BertTokenizer
import torch

pre_train_model_name =  "bert-base-uncased"
sentence = "MachineLEarning engineers live in California"

model = BertModel.from_pretrained(pre_train_model_name)
tokenizer = BertTokenizer.from_pretrained(pre_train_model_name)
tokens = tokenizer.tokenize(sentence)
print("Input sentence ", sentence)
print("Input tokens ", tokens)
