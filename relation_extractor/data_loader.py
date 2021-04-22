import pickle as pickle
from typing import Dict

import pandas as pd
import torch
from pandas import ExcelWriter
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


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
    print(f"Tokenizing...{tokenizer}")
    tokenized_raw = tokenize_dataset(data_raw, tokenizer)

    return RE_Dataset(tokenized_raw, data_raw["label"].values), None


def load_k_fold_train_dataset(data_root: str, tokenizer: str, k: int=5, seed: int=None):
    print("Loading dataset...")
    with open(f"{data_root}/label_type.pkl", 'rb') as f:
        label_type: Dict = pickle.load(f)
        label_type["blind"] = 100
    
    # 주의: new type만 읽을 수 있음
    data_raw = pd.read_excel(f"{data_root}/train_new/train_new.xlsx", "combined_all")
    labels = []
    sentences = []
    
    # 엔티티 추가
    for index, row in enumerate(data_raw.iloc):
        text: str = row["context"]
        if text.find("{{ sbj }}") == -1 or text.find("{{ obj }}") == -1:
            raise Exception(f"{index} row data corrupted\n\n{row}")
        
        labels.append(label_type[row["label"]])
        text = text.replace("{{ sbj }}", str(row["sbj_entity"]))
        text = text.replace("{{ obj }}", str(row["obj_entity"]))
        sentences.append(text)
    
    data_raw = pd.DataFrame(
        {
            "sentence": sentences, 
            "entity_01": data_raw["sbj_entity"].astype(str), 
            "entity_02": data_raw["obj_entity"].astype(str), 
            "label": labels
        }
    )

    print("Splitting...")
    if k > 1:
        k_fold_splitter = StratifiedKFold(n_splits=k, shuffle=(seed != None), random_state=seed)
        k_fold_data_indexes = k_fold_splitter.split(X=data_raw, y=data_raw["label"])

        k_fold_raws = []
        for train_indexes, valid_indexes in k_fold_data_indexes:
            train_raw: pd.DataFrame = data_raw.iloc[train_indexes]
            valid_raw: pd.DataFrame = data_raw.iloc[valid_indexes]
            
            train_raw.reset_index(drop=True)
            valid_raw.reset_index(drop=True)
            k_fold_raws.append((train_raw, valid_raw))
        
        # K-Fold 확을 위한 저장용
        with ExcelWriter(f"./results/kfold_results.xlsx", engine="xlsxwriter") as writer:
            print("Saving...")
            for index, raw_group in enumerate(k_fold_raws):
                raw_group[0].to_excel(writer, f"Train {index}", index=False)
                raw_group[1].to_excel(writer, f"Valid {index}", index=False)
            
            writer.save()
    else:
        train_raw = data_raw    # 구현하다 말음 주의

    print(f"Tokenizing by {tokenizer}...")
    train_set = []
    valid_set = []
    for raw in k_fold_raws:
        tokenized_train = tokenize_dataset(raw[0], tokenizer)
        tokenized_valid = tokenize_dataset(raw[1], tokenizer)
        train_set.append(RE_Dataset(tokenized_train, raw[0]["label"].values))
        valid_set.append(RE_Dataset(tokenized_valid, raw[1]["label"].values))
    
    print("Ready to serve!")
    return train_set, valid_set



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


if __name__ == "__main__":
    # Test K-Fold
    load_k_fold_train_dataset("/opt/ml/input/data", "kykim/bert-kor-base")
