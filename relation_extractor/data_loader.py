import pickle as pickle
from typing import Dict
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch


class REDataset(Dataset):
    def __init__(self, raw: pd.DataFrame) -> None:
        self.raw_data: pd.DataFrame = raw
        self.targets = [x for x in range(len(raw))]
    
    def __getitem__(self, idx):
        return self.targets[idx]

    def __len__(self):
        return len(self.targets)


def load_dataset(data_root: str, tokenizer: str, data_type="train") -> None:
    """데이터로부터 데이터셋 생성

    Args:
        data_root (str): 원본 데이터들(Train, Test, ..., etc)이 들어 있는 최상위 폴더
        data_type (str, optional): 최상위 폴더에서 실제 데이터 하위 폴더명(tsv 파일이 있는 폴더). Defaults to "train".

    Returns:
        [type]: 데이터셋 객체
    """
    # label_type, classes
    with open(f"{data_root}/label_type.pkl", 'rb') as f:
        label_type: Dict = pickle.load(f)
        label_type["blind"] = 100

    data_raw: pd.DataFrame = None
    # load dataset
    if data_type == "train":
        data_raw = pd.read_csv(f"{data_root}/{data_type}/{data_type}.tsv", delimiter='\t', header=None)
        labels = []
        for label_raw in data_raw[8]:
            labels.append(label_type[label_raw])
        data_raw = pd.DataFrame(
            {
                "sentence": data_raw[1], 
                "entity_01": data_raw[2], 
                "entity_02": data_raw[5], 
                "label": labels
            }
        )
    elif data_type == "train_new":
        data_raw = pd.read_excel(f"{data_root}/{data_type}/{data_type}.xlsx", "combined_all")
        labels = []
        sentences = []
        for row in data_raw.iloc:
            labels.append(label_type[row["label"]])
            text = row["context"].replace("{{ sbj }}", row["sbj_entity"])
            text = text.replace("{{ obj }}", row["obj_entity"])
            sentences.append(text)
        
        data_raw = pd.DataFrame(
            {
                "sentence": sentences, 
                "entity_01": data_raw["sbj_entity"], 
                "entity_02": data_raw["obj_entity"], 
                "label": labels
            }
        )
    else:
        raise Exception("No such dataset type")    

    # validation dataset으로 분리하려면 여기서부터 코드를 집어 넣으면 됨
    print(data_raw)
    tokenized_raw = tokenize_dataset(data_raw, tokenizer)

    return RE_Dataset(tokenized_raw, data_raw["label"].values)


# Baseline codes

class RE_Dataset(Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenize_dataset(dataset: pd.DataFrame, pretrained_model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

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
        max_length=256,
        add_special_tokens=True,
    )

    return tokenized_sentences