import os
import random
from typing import Dict, List, Union

import numpy as np
import torch
import yaml

from relation_extractor.data_loader import load_dataset, load_k_fold_train_dataset
from relation_extractor.predictor import predict, predict_fold_enssemble


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

        # 옛날 방식과 호환이 안 된다...이럴 거면 뭐하러 프로그램 구조를 이렇게 짰나...ㅠㅠ
        # train_dataset, valid_dataset = load_dataset(data_root=config["data_root"], tokenizer=model_info["name"], data_type="train")
        train_dataset, valid_dataset = load_k_fold_train_dataset(data_root=config["data_root"], tokenizer=model_info["name"], seed=seed)


        # Base Model 이름, 타입 yaml 설정 읽기
        trainee_class = getattr(trainer, trainee_setting["trainee_type"])
        trainee: trainer.TraineeBase = trainee_class(
            device=device, 
            train_dataset=train_dataset, valid_dataset=valid_dataset, 
            **trainee_setting
        )

        trainee.train(
            config["checkpoints_path"], 
            config["tensorboard_log_path"], 
            config["train_log_path"]
        ) # Training
        # predict(f"./results/checkpoint/{trainee.name}/last_checkpoint/", model_info["name"], model_info["type"], device)
        predict_fold_enssemble(
            f"./results/checkpoint/{trainee.name}",
            model_info["name"], 
            model_info["type"], 
            device
        )


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
    # predict_fold_enssemble(f"./results/checkpoint/kor-bert-k-fold/last_checkpoint", "kykim/bert-kor-base", "Bert", device)

    # predict_fold_enssemble(
    #     f"./results/checkpoint/xlm-roberta-fold-origin",
    #     "xlm-roberta-large", 
    #     "XLMRoberta", 
    #     device
    # )
    
    # predict(
    #     f"./results/checkpoint/xlm-roberta-one/last_checkpoint",
    #     "xlm-roberta-large", 
    #     "XLMRoberta", 
    #     device
    # )

# 5개중 가장 성능 높은 것 제출
# xlm-???? 적용 (앙상블 하지 말고)
# 앙상블 제출 (소프트 보팅)
# 클래스 별 모델 정확도 측정 -> validation을 다시 수행해서 클래스 별로 맞추고 못 맞춘 것을 체크
# ==> 독립성 검정도 할 수 있을까? ==> 모델 별 클래스 추론이 명확히 차이가 난다면 해당 클래스의 모델 별 가중치를 달리하여 적용
# 시간이 되면 xlm 앙상블 적용