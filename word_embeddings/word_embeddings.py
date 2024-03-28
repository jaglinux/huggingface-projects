from transformers import BertModel, BertTokenizer
import torch

pre_train_model_name =  "bert-base-uncased"
sentence = "MachineLEarning engineers live in California"

model = BertModel.from_pretrained(pre_train_model_name)
tokenizer = BertTokenizer.from_pretrained(pre_train_model_name)
tokens = tokenizer.tokenize(sentence)
print("Input sentence ", sentence)
print("Input tokens ", tokens)
tokens.insert(0, '[CLS]')
tokens.append('[SEP]')
print("Input tokens after classification and sentence pair ", tokens)
print('Len of tokens ', len(tokens))
# while len(tokens) < 16:
#     tokens.append('[PAD]')
tokens += ['[PAD]'] * (16 - len(tokens))
print("Input tokens after padding ", tokens)
attention_mask = [1 if i != '[PAD]' else 0 for i in tokens]
print("attention_mask ", attention_mask)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print("token ids ", token_ids)
token_ids = torch.tensor(token_ids).unsqueeze(0)
print(token_ids, token_ids.shape, token_ids.dim())
attention_mask = torch.tensor(attention_mask).unsqueeze(0)
print(attention_mask, attention_mask.shape, attention_mask.dim())

output = model(token_ids, attention_mask=attention_mask)

print("model ouput ", output)
last_hidden_layer = output[0]
print("model output last hidden layer ", last_hidden_layer, last_hidden_layer.shape, last_hidden_layer.dim())