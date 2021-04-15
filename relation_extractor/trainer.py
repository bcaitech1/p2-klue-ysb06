import os
from datetime import datetime

import pytz
import torch
import torch.nn
import transformers
import yaml
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from transformers import (Trainer, TrainerCallback, TrainerControl,
                          TrainerState, TrainingArguments)


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

        # After training


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
    }


class TrainingEndCallback(TrainerCallback):
    def __init__(self, trainee_name: str, hyperparameters: dict, log_path: str) -> None:
        self.trainee_name = trainee_name
        self.hyperparameters = hyperparameters
        self.target_path = log_path

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

        with open(f"{self.target_path}/{self.trainee_name}/{datetime.now(self.kst).strftime('%Y%m%d_%H%M%S')}.yaml", 'w') as fw:
            yaml.dump(training_result, fw)
        return super().on_train_end(args, state, control, **kwargs)
