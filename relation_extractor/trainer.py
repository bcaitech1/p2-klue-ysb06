import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from relation_extractor import data_loader


class TraineeBase:
    def __init__(
        self,
        device: torch.device,
        train_dataset: Dataset,
        valid_dataset: Dataset
    ) -> None:

        self.device: torch.device = device
        self.train_set = train_dataset
        self.valid_set = valid_dataset

    def train(self, result_path: str, log_path: str):
        print("Training not defined")


class BaselineTrainee(TraineeBase):
    def __init__(
            self, 
            device: torch.device, 
            train_dataset: Dataset, 
            valid_dataset: Dataset = None, 
            model_name: str = "bert-base-multilingual-cased",
            num_epochs = 4
        ) -> None:
        super().__init__(device, train_dataset, valid_dataset)

        self.model_name = model_name
        self.num_epochs = num_epochs

    def train(self, result_path: str, log_path: str):
        # load model and tokenizer
        MODEL_NAME = self.model_name

        # load dataset
        RE_train_dataset = self.train_set

        # setting model hyperparameter
        bert_config = BertConfig.from_pretrained(MODEL_NAME)
        bert_config.num_labels = 42
        model = BertForSequenceClassification(bert_config)
        model.parameters
        model.to(self.device)

        # 사용한 option 외에도 다양한 option들이 있습니다.
        # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
        training_args = TrainingArguments(
            output_dir=result_path,          # output directory
            save_total_limit=3,              # number of total save model.
            save_steps=500,                 # model saving step.
            num_train_epochs=4,              # total number of training epochs
            learning_rate=5e-5,               # learning_rate
            per_device_train_batch_size=16,  # batch size per device during training
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir=log_path,            # directory for storing logs
            logging_steps=100,              # log saving step.
        )
        trainer = Trainer(
            # the instantiated 🤗 Transformers model to be trained
            model=model,
            args=training_args,                  # training arguments, defined above
            train_dataset=RE_train_dataset,         # training dataset
        )

        # train model
        trainer.train()


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
    }
