# import os
# from transformers import BertModel, BertTokenizer, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
# import torch

# from src.scripts.pipeline import load_configuration

# # Set the TOKENIZERS_PARALLELISM environment variable to false
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def count_heads(model):
#     remaining_heads = {}
#     for i, layer in enumerate(model.bert.encoder.layer):
#         remaining_heads[i] = layer.attention.self.num_attention_heads
#     return remaining_heads

# # Load a pretrained BERT model and tokenizer

# model_name_or_path = f"/home/dbekris/src/tuning/mrpc"
# max_seq_length=128
# heads_output_dir = f"/home/dbekris/src/pruning_heads/mrpc"
# criterion = "mi"
# output_mode = "classification"
# world_size = 2

# # Load model, tokenizer
# model, dataset, cls_id, sep_id, tokenizer = load_configuration(model_name_or_path, "mrpc", max_seq_length)

# # Count parameters before pruning
# params_before_pruning = count_parameters(model)
# print(f'Number of parameters before pruning: {params_before_pruning}')

# # Count heads before pruning
# heads_before_pruning = count_heads(model)
# print(f'Heads before pruning: {heads_before_pruning}')

# # Example sentence
# sentence = "This is a sample sentence to demonstrate pruning heads."

# # Tokenize the sentence
# inputs = tokenizer(sentence, return_tensors='pt')

# inputs["input_ids"] = inputs["input_ids"].cuda(0)
# inputs["token_type_ids"] = inputs["token_type_ids"].cuda(0)
# inputs["attention_mask"] = inputs["attention_mask"].cuda(0)

# # Prune attention heads
# # The `heads_to_prune` dictionary specifies which heads to prune for each layer.
# # For example, {layer_index: [head_indices]} means prune heads at the specified layer index.
# import numpy as np
# head_mask = np.load("/home/dbekris/src/pruning_heads/mrpc/head_mask.npy")
# head_mask = torch.from_numpy(head_mask)

# model = model.cuda(0)
# heads_to_prune = {}
# for layer in range(len(head_mask)):
#     heads_to_mask = [h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()]
#     heads_to_prune[layer] = heads_to_mask

# print(heads_to_prune)

# # heads_to_prune = {
# #     0: [0, 1],  # Prune heads 0 and 1 in layer 0
# #     1: [2, 3],  # Prune heads 2 and 3 in layer 1
# #     # Add more layers and heads as needed
# # }

# heads_to_prune = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [11], 9: [], 10: [6], 11: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}

# try:
#     # Prune the heads
#     model.prune_heads(heads_to_prune)
#     print("Heads pruned successfully.")
# except IndexError as e:
#     print(f"An error occurred while pruning heads: {e}")

# # # Count parameters after pruning
# # params_after_pruning = count_parameters(model)
# # print(f'Number of parameters after pruning: {params_after_pruning}')

# # Count heads after pruning
# heads_after_pruning = count_heads(model)
# print(f'Heads after pruning: {heads_after_pruning}')

# # Model output after pruning (optional)
# pruned_output = model(**inputs)

# # Print the outputs for comparison (optional)
# # print("Original Output:", original_output.last_hidden_state)
# # print("Pruned Output:", pruned_output.last_hidden_state)

import torch

# labels = []

# label_list = [
#     torch.tensor([
#         [1],
#         [0],
#         [0],
#         [1]]),
#     torch.tensor([
#         [1],
#         [0],
#         [1],
#         [0]])
#     ]

# for label in label_list:
#     labels.append(label)

# cat_label = torch.cat(labels, dim=0)

# print(cat_label)
# #####################
# preds = []

# pred_list = [
#     torch.tensor([
#         [0.1, 0.9],
#         [0.7, 0.3],
#         [0.2, 0.8],
#         [0.1, 0.9],]),
#     torch.tensor([
#         [0.1, 0.9],
#         [0.7, 0.3],
#         [0.2, 0.8],
#         [0.2, 0.8],])
#     ]

# for pred in pred_list:
#     preds.append(pred)

# cat_pred = torch.cat(preds, dim=0)

# print(cat_pred)
# import numpy as np
# preds_max = np.argmax(cat_pred, axis=1)

# print(preds_max)

# assert len(preds_max) == len(cat_label)

# from sklearn.metrics import matthews_corrcoef, f1_score
# def simple_accuracy(preds, labels):
#     return (preds == labels).float().mean()

# def acc_and_f1(preds, labels):
#     acc = simple_accuracy(preds, labels)
#     f1 = f1_score(y_true=labels, y_pred=preds)
#     return {
#         "acc": acc,
#         "f1": f1,
#         "acc_and_f1": (acc + f1) / 2,
#     }
# print(simple_accuracy(preds_max, cat_label))

# b_preds = torch.load("../scripts/b_preds.pt")
# preds = torch.load("../scripts/preds.pt")
# labels = torch.load("../scripts/labels.pt")

# print(b_preds.shape)
# print(preds.shape)
# print(labels.shape)

# print(b_preds)
# print(preds)
# print(labels)

# a = ['hello', 'you', 'there']
# b = [('hello'), ('you'), ('there')]
# c = [('hello', ), ('you', ), ('there', )]
# d = [('hello', 'father'), ('you', 'little'), ('there', 'son of a bitch')]


# e = [(('hello')), (('you')), (('there'))]
# f = [(('hello', )), (('you', )), (('there', ))]
# g = [(('hello', 'father')), (('you', 'little')), (('there', 'son of a bitch'))]

# print(a, type(a))
# print(b, type(b))
# print(c, type(c))
# print(d, type(d))
# print(e, type(e))
# print(f, type(f))
# print(g, type(g))

# print(*a, type(*a))
# print(*b, type(*b))
# print(*c, type(*c))
# print(*d, type(*d))
# print(*e, type(*e))
# print(*f, type(*f))
# print(*g, type(*g))

atts = torch.load("/home/dbekris/src/thesis-layerconductance-structured-pruning-bert/src/scripts/atts.pth")
attrs = torch.load("/home/dbekris/src/thesis-layerconductance-structured-pruning-bert/src/scripts/attrs.pth")

print(len(atts))
print(atts[0].shape)

var_1 = atts[1].view(2, 12, 12, -1)[0][0][0]
var_2 = attrs[1].view(2, 12, 12, -1)[0][0][0]

import matplotlib.pyplot as plt

plt.scatter(var_1, var_2)
plt.savefig("scatter.png")