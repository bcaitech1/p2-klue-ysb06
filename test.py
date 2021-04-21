# from numpy import mod
# import torch.nn
# from transformers import BertConfig, BertForSequenceClassification

# config = BertConfig.from_pretrained("kykim/bert-kor-base")
# config.num_labels = 42
# model: torch.nn.Module = BertForSequenceClassification.from_pretrained("kykim/bert-kor-base", config=config)

# for name, param in model.named_parameters():
#     if "output" in name or "classifier" in name or "bert.pooler.dense" in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False 

# for name, param in model.named_parameters():
#     print(f"{name}: {param.requires_grad}")

# print(model)



import numpy as np
# from sklearn.model_selection import StratifiedKFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# y = np.array([0, 1, 1, 2])
# skf = StratifiedKFold(n_splits=2)

# StratifiedKFold(n_splits=2, random_state=None, shuffle=False)

# for train_index, test_index in skf.split(X, y):
#     print("TRAIN:", train_index, "TEST:", test_index)

print(X)