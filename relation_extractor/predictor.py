import pickle as pickle

import numpy as np
import pandas as pd
import torch
import transformers
from torch.nn import Module
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from dataset_hub.analyzer import refine_special_letter


def predict(model_path: str, model_name: str, model_type: str, target_device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = getattr(
        transformers, 
        model_type + "Config").from_pretrained(model_path)
    config.num_labels = 42

    model: Module = getattr(
        transformers, 
        model_type + "ForSequenceClassification").from_pretrained(model_path, config=config)
    model = model.to(target_device)
    model.eval()

    # load test datset
    test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    test_dataset = TestDataset(test_dataset ,test_label)

    # Inference
    dataloader = DataLoader(test_dataset, batch_size=40, shuffle=False)    
    output_pred = []

    for i, data in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(target_device),
                attention_mask=data['attention_mask'].to(target_device),
                # token_type_ids=data['token_type_ids'].to(target_device)
            )
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)

    # Inference End
    pred_answer = np.array(output_pred).flatten()

    output = pd.DataFrame(pred_answer, columns=['pred'])
    output.to_csv('./results/submission.csv', index=False)
    print("Inference Finished!")


def predict_fold_enssemble(model_path: str, model_name: str, model_type: str, target_device: torch.device, fold: int=5):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    models = []

    for n in range(fold):
        config = getattr(
            transformers, 
            model_type + "Config").from_pretrained(f"{model_path}/last_checkpoint/{n}")
        config.num_labels = 42

        model: Module = getattr(
            transformers, 
            model_type + "ForSequenceClassification").from_pretrained(f"{model_path}/last_checkpoint/{n}", config=config)
        model = model.to(target_device)
        model.eval()
        models.append(model)

    # load test datset
    test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    test_dataset = TestDataset(test_dataset ,test_label)

    # Inference
    dataloader = DataLoader(test_dataset, batch_size=40, shuffle=False)    
    output_pred = []

    result_list = [[] for _ in range(len(models))]
    for data in tqdm(dataloader):
        with torch.no_grad():
            for index, model in enumerate(models):
                output = model(
                    input_ids=data['input_ids'].to(target_device),
                    attention_mask=data['attention_mask'].to(target_device),
                    # token_type_ids=data['token_type_ids'].to(target_device)
                )

                logit: torch.Tensor = output[0]
                logit = logit.detach().cpu().numpy()
                result = np.argmax(logit, axis=-1)

                result_list[index].append(result.tolist())

    # Inference End
    output_df = pd.DataFrame()
    for n in range(len(result_list)):
        output_df[f"pred{n}"] = np.array(result_list[n]).flatten()

    # output = pd.DataFrame(pred_answer, columns=['pred'])
    # output.to_csv('./results/submission.csv', index=False)
    output_df = output_df.mode(axis=1)[0].to_frame()
    output_df.columns = ["pred"]
    print(output_df)
    # output_df.to_excel("./results/test.xlsx", engine="xlsxwriter")
    output_df.to_csv('./results/submission.csv', index=False)
    print("Inference Finished!")


def load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir)
    test_label = test_dataset['label'].values
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label

# Dataset ??????.
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels) 

# ?????? ????????? tsv ????????? ????????? ????????? DataFrame?????? ?????? ???????????????.
# ????????? DataFrame ????????? baseline code description ???????????? ??????????????????.


def preprocessing_dataset(dataset, label_type):
    label = []
    for i in dataset[8]:
        if i == 'blind':
            label.append(100)
        else:
            label.append(label_type[i])
    out_dataset = pd.DataFrame({
        'sentence': dataset[1], 
        'entity_01': dataset[2], 
        'entity_02': dataset[5], 
        'label': label, 
    })

    # refine_special_letter(out_dataset, target_column="sentence", print_not_refined=True)

    return out_dataset

# tsv ????????? ???????????????.


def load_data(dataset_dir):
    # load label_type, classes
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    # load dataset
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    # preprecessing dataset
    dataset = preprocessing_dataset(dataset, label_type)

    return dataset

# bert input??? ?????? tokenizing.
# tip! ????????? ????????? tokenizer??? special token?????? ???????????? ???????????? ????????? ????????? ?????? ??? ????????????.
# baseline code????????? 2?????? ????????? ??????????????????.


def tokenized_dataset(dataset, tokenizer):
    concat_entity = []

    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)

    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=100,
        add_special_tokens=True,
    )
    return tokenized_sentences


if __name__ == "__main__":
    # Search target device
    print(f"PyTorch version: [{torch.__version__}]")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  Target device: [{device}]")

    name = "kor-bert-new-data"

    predict(f"./results/checkpoint/{name}/last_checkpoint", "Bert", device)