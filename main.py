import os
import random
from typing import Dict, List, Union

import numpy as np
import torch
import yaml

from relation_extractor.data_loader import load_dataset
from relation_extractor.predictor import predict


def initialize():
    training_config = {
        "data_root": "/opt/ml/input/data",
        "checkpoints_path": "./results/checkpoint",
        "tensorboard_log_path": "./results/tensorboard",
        "train_log_path": "./results/log",
        "trainees": [
            {
                "trainee_type": "BaselineTrainee",
                "trainee_name": "baseline_trainee",
                "hyperparameters": {
                    "model": {
                        "name" : "bert-base-multilingual-cased",
                        "type" : "Bert",
                    },
                    "args": {
                        "num_train_epochs": 5,              # total number of training epochs
                        "learning_rate": 5e-5,              # learning_rate
                        "per_device_train_batch_size": 16,  # batch size per device during training
                        "warmup_steps": 500,                # number of warmup steps for learning rate scheduler
                        "weight_decay": 0.01,               # strength of weight decay
                    },
                    "seed": 327459
                }
            },
        ]
    }

    with open(f"./config.yaml", 'w') as fw:
        yaml.dump(training_config, fw)

    return training_config


def load_config():
    training_config = {}
    with open(f"./config.yaml", 'r') as fr:
        training_config = yaml.load(fr, Loader=yaml.FullLoader)

    return training_config


def run_pipeline(target: Union[int, None]):
    from relation_extractor import trainer

    # Search target device
    print(f"PyTorch version: [{torch.__version__}]")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  Target device: [{device}]")

    # Load config file
    config = {}
    if not os.path.isfile("./config.yaml"):
        config = initialize()   # 샘플 config.yaml 생성
        print("initialized!")
    else:
        config = load_config()

    if not os.path.isdir(config["checkpoints_path"]):
        os.mkdir(config["checkpoints_path"])

    if not os.path.isdir(config["tensorboard_log_path"]):
        os.mkdir(config["tensorboard_log_path"])

    if not os.path.isdir(config["train_log_path"]):
        os.mkdir(config["train_log_path"])

    # ----- Load and train trainee
    trainee_settings: List[Dict] = config["trainees"]

    for index, trainee_setting in enumerate(trainee_settings):
        if target != None:
            if index != target:
                continue

        trainee_name = trainee_setting["trainee_name"]
        print(f"Training {trainee_name}...")
        # trainee_type = trainee_setting["trainee_type"]
        model_info = trainee_setting["hyperparameters"]["model"]
        # args = trainee_setting["hyperparameters"]["args"]
        seed = trainee_setting["hyperparameters"]["seed"]
        
        # Seed 설정
        seed_everything(seed)

        train_dataset = load_dataset(data_root=config["data_root"], tokenizer=model_info["name"], data_type="train_new")
        valid_dataset = None    # 추후 데이터 늘리고 Valid Set 만들 것

        # Base Model 이름, 타입 yaml 설정 읽기
        trainee_class = getattr(trainer, trainee_setting["trainee_type"])
        trainee: trainer.TraineeBase = trainee_class(
            device=device, 
            train_dataset=train_dataset, valid_dataset=valid_dataset, 
            **trainee_setting
        )

        trainee.train(config["checkpoints_path"], config["tensorboard_log_path"], config["train_log_path"]) # Training
        predict(f"./results/checkpoint/{trainee.name}/last_checkpoint/", "Bert", device)


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    run_pipeline(None)

    # print(f"PyTorch version: [{torch.__version__}]")
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(f"  Target device: [{device}]")

    # name = "kor-bert-new-data"

    # predict(f"./results/checkpoint/{name}/last_checkpoint", "Bert", device)
