import pickle as pickle

import numpy as np
import pandas as pd
import torch
import transformers
from torch.nn import Module
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset_hub.analyzer import refine_special_letter


def predict(model_path: str, model_type: str, target_device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained("kykim/bert-kor-base")

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
                token_type_ids=data['token_type_ids'].to(target_device)
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


def load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir)
    test_label = test_dataset['label'].values
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label

# Dataset 구성.
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

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.


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

    refine_special_letter(out_dataset, target_column="sentence", print_not_refined=True)

    return out_dataset

# tsv 파일을 불러옵니다.


def load_data(dataset_dir):
    # load label_type, classes
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    # load dataset
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    # preprecessing dataset
    dataset = preprocessing_dataset(dataset, label_type)

    return dataset

# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.


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