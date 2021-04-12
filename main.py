from relation_extractor.data_loader import load_dataset
import torch
import os
import yaml


def initialize():
    training_config = {
        "data_root": "./data",
        "results_path": "./results",
        "tensorboard_log_path": "./results/tensorboard",
        "target": -1,
        "settings": [
            {
                "trainee_type": "BaselineTrainee",
                "trainee_name": "sample_name",
                "hyperparameters": {
                    "num_epochs": 5
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


def main():
    from relation_extractor import trainer

    # Search target device
    print(f"PyTorch version: [{torch.__version__}]")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  Target device: [{device}]")

    # Load config file
    config = {}
    if not os.path.isfile("./config.yaml"):
        config = initialize()
    else:
        config = load_config()

    if not os.path.isdir(config["results_path"]):
        os.mkdir(config["results_path"])

    if not os.path.isdir(config["tensorboard_log_path"]):
        os.mkdir(config["tensorboard_log_path"])

    # ----- Load and train trainee
    training_setting = config["settings"][int(config["target"])]

    # Load dataset
    train_dataset = load_dataset("./data")
    valid_dataset = None

    # Load trainee
    trainee_class = getattr(trainer, training_setting["trainee_type"])
    trainee: trainer.TraineeBase = trainee_class(
        device=device, 
        train_dataset=train_dataset, valid_dataset=valid_dataset, 
        **training_setting["hyperparameters"]
    )
    trainee.train() # Training


if __name__ == "__main__":
    main()
