"""
Task E-c: Emotion Classification (multi-label classification)

Input:
 - a tweet

Task: classify the tweet as 'neutral or no emotion' or as one or more
of eleven given emotions that best represent the mental state of the tweeter:

    - anger (also includes annoyance and rage) can be inferred
    - anticipation (also includes interest and vigilance) can be inferred
    - disgust (also includes disinterest, dislike and loathing) can be inferred
    - fear (also includes apprehension, anxiety, concern, and terror) can be inferred
    - joy (also includes serenity and ecstasy) can be inferred
    - love (also includes affection) can be inferred
    - optimism (also includes hopefulness and confidence) can be inferred
    - pessimism (also includes cynicism and lack of confidence) can be inferred
    - sadness (also includes pensiveness and grief) can be inferred
    - surprise (also includes distraction and amazement) can be inferred
    - trust (also includes acceptance, liking, and admiration) can be inferred
    - neutral or no emotion

"""

from lib.config import MODEL_EC
from lib.utils import parse
from trainer import define_trainer, model_training

def train_e_c(finetune=True, unfreeze=0):
    model_config = MODEL_EC

    # loading datasets (train/development/test)
    X_train, y_train = parse(dataset="train")
    X_dev, y_dev = parse(dataset="dev")
    X_test, y_test = parse(dataset="gold")

    datasets = {
        "train": (X_train, y_train),
        "dev": (X_dev, y_dev),
        "test": (X_test, y_test),
    }

    name = model_config["name"]
    trainer = define_trainer("mclf", config=model_config, name=name,
                             datasets=datasets,
                             monitor="dev",
                             finetune=finetune)

    model_training(trainer, model_config["epochs"], checkpoint=True)

    desc = "Train E-c"
    trainer.log_training(name, desc)


def main():
    finetune = True
    train_e_c(finetune=finetune)

if __name__ == '__main__':
    main()