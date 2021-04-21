import os
from datetime import datetime
from typing import List

import pytz
import torch
import torch.nn
import transformers
import yaml
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import Dataset
from transformers import (Trainer, TrainerCallback, TrainerControl,
                          TrainerState, TrainingArguments, AutoTokenizer)
                          


class TraineeBase:
    def __init__(
        self,
        device: torch.device,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        trainee_name: str,
        trainee_type: str,
        hyperparameters: dict
    ) -> None:

        self.device: torch.device = device
        self.train_set = train_dataset
        self.valid_set = valid_dataset
        self.name = trainee_name
        self.hyperparameters = hyperparameters

    def train(self, result_path: str, tensorboard_path: str, log_path: str):
        print("Training not defined")


class BaselineTrainee(TraineeBase):
    def __init__(
            self,
            device: torch.device,
            train_dataset: Dataset,
            valid_dataset: Dataset,
            trainee_name: str,
            trainee_type: str,
            hyperparameters: dict
        ) -> None:
        super().__init__(device, train_dataset, valid_dataset, trainee_name, trainee_type, hyperparameters)

    def train(self, result_path: str, tensorboard_path: str, log_path: str):
        # load dataset
        RE_train_dataset = self.train_set

        # hyperparameters
        model_info = self.hyperparameters["model"]
        args = self.hyperparameters["args"]

        # setting model hyperparameter
        config = getattr(transformers, model_info["type"] + "Config").from_pretrained(model_info["name"])
        config.num_labels = 42

        model: torch.nn.Module = getattr(transformers, model_info["type"] + "ForSequenceClassification").from_pretrained(model_info["name"], config=config)
        model.to(self.device)

        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 사용한 option 외에도 다양한 option들이 있습니다.
        # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
        training_args = TrainingArguments(
            output_dir=f"{result_path}/{self.name}",    # output directory
            save_total_limit=3,                         # number of total save model.
            save_steps=500,                             # model saving step.
            logging_dir=f"{tensorboard_path}/{self.name}",      # directory for storing logs
            logging_steps=200,                          # log saving step.
            **args
        )

        trainer = Trainer(
            # the instantiated 🤗 Transformers model to be trained
            model=model,
            args=training_args,             # training arguments, defined above
            train_dataset=RE_train_dataset, # training dataset
            callbacks=[TrainingEndCallback(self.name, self.hyperparameters, log_path)]
        )

        # train model
        trainer.train()
        trainer.save_model(f"./results/checkpoint/{self.name}/last_checkpoint/")

        # After training


class BaselineKFoldTrainee(TraineeBase):
    def __init__(
            self,
            device: torch.device,
            train_dataset: List[Dataset],
            valid_dataset: List[Dataset],
            trainee_name: str,
            trainee_type: str,
            hyperparameters: dict
        ) -> None:
        # 주의 super 클래스(TraineeBase)에 넣는 Dataset 인자의 형식이 맞지 않음. (기능 상 문제는 없지만 Base 재설계가 필요)
        super().__init__(device, train_dataset, valid_dataset, trainee_name, trainee_type, hyperparameters)
    
    def train(self, result_path: str, tensorboard_path: str, log_path: str):
        # hyperparameters
        model_type = self.hyperparameters["model"]["type"]
        model_name = self.hyperparameters["model"]["name"]
        args = self.hyperparameters["args"]

        # 임시 코드, 추후 Loader에서 토크나이저 받기
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for index, (train_dataset, valid_dataset) in enumerate(zip(self.train_set, self.valid_set)):
            if not os.path.isdir(f"{log_path}/{self.name}"):
                os.mkdir(f"{log_path}/{self.name}")
            
            if not os.path.isdir(f"{log_path}/{self.name}/{index}"):
                os.mkdir(f"{log_path}/{self.name}/{index}")
            
            print(f"Fold {index} training...")
            print()

            # config는 공용으로 써도 될 거 같은데
            # 일단은 매 Fold마다 생성하는 걸로
            config = getattr(transformers, f"{model_type}Config").from_pretrained(model_name)
            config.num_labels = 42

            model: torch.nn.Module = getattr(transformers, f"{model_type}ForSequenceClassification").from_pretrained(model_name, config=config)
            model.to(self.device)

            # 모델 Freeze는 성능이 나오지 않아서 적용 안 함.
            # 아마도 모델의 용도가 달라서 그런 것으로 추측 (Fill-mask <> Relation Extraction)

            # Set training env. and hyperparameters
            training_args: TrainingArguments = TrainingArguments(
                output_dir=f"{result_path}/{self.name}/{index}",    # output directory
                save_total_limit=3,                         # number of total save model.
                save_steps=500,                             # model saving step. Best 저장 정책이므로 무시됨.
                logging_dir=f"{tensorboard_path}/{self.name}/{index}",      # directory for storing logs
                logging_steps=100,                          # log saving step.
                logging_first_step=True,
                evaluation_strategy="epoch",
                metric_for_best_model="accuracy",
                load_best_model_at_end=True,
                **args
            )

            trainer: Trainer = Trainer(
                # the instantiated 🤗 Transformers model to be trained
                model=model,
                tokenizer=tokenizer,
                args=training_args,             # training arguments, defined above
                train_dataset=train_dataset, # training dataset
                eval_dataset=valid_dataset,
                compute_metrics=compute_metrics,
                callbacks=[TrainingEndCallback(f"{self.name}", self.hyperparameters, log_path, sub_id=index)],
            )
            # 전체 횟수와 lr 줄어드는 부분 Tensorboard로 체크하고 cosine annealing optimizer 로 변경

            # train model
            trainer.train()
            trainer.save_model(f"./results/checkpoint/{self.name}/last_checkpoint/{index}/")
            print()
            print(f"Fold {index} training complete.")
            print()


class TrainingEndCallback(TrainerCallback):
    def __init__(self, trainee_name: str, hyperparameters: dict, log_path: str, sub_id: int = -1) -> None:
        self.trainee_name = trainee_name
        self.hyperparameters = hyperparameters
        self.target_path = log_path
        self.trainee_id = sub_id

        self.kst = pytz.timezone('Asia/Seoul')
        self.start_time = datetime.now(self.kst).strftime("%Y-%m-%d %H:%M:%S")
        
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        epoch = 0
        summarized_history = []
        for log in state.log_history:
            if log["epoch"] >= epoch:
                epoch += 1
                summarized_history.append(log)
        summarized_history.append(state.log_history[-1])

        training_result = {
            "training_start_time": self.start_time,
            "training_end_time": datetime.now(self.kst).strftime("%Y-%m-%d %H:%M:%S"),
            "trainee_name": self.trainee_name,
            "hyperparameters": self.hyperparameters,
            "state": {
                "history": summarized_history,
                "best_metric": state.best_metric,
                "best_checkpoint": state.best_model_checkpoint
            }
        }

        if not os.path.isdir(f"{self.target_path}/{self.trainee_name}"):
            os.mkdir(f"{self.target_path}/{self.trainee_name}")

        if self.trainee_id >= 0:
            target_path = f"{self.target_path}/{self.trainee_name}/{self.trainee_id}"
            if not os.path.isdir(target_path):
                os.mkdir(target_path)

            with open(f"{target_path}/{datetime.now(self.kst).strftime('%Y%m%d_%H%M%S')}.yaml", 'w') as fw:
                yaml.dump(training_result, fw)
        else:
            with open(f"{self.target_path}/{self.trainee_name}/{datetime.now(self.kst).strftime('%Y%m%d_%H%M%S')}.yaml", 'w') as fw:
                yaml.dump(training_result, fw)
        
        print(state.log_history)
        return super().on_train_end(args, state, control, **kwargs)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
