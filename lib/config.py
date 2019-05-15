import os
import torch

# device settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on: {}".format(DEVICE))

# path settings
BASE_PATH = "/home/houyu/learning/FinalProject"
MODEL_PATH = os.path.join(BASE_PATH, "out/model")
EXPS_PATH = os.path.join(BASE_PATH, "out/experiments")
# ATT_PATH = os.path.join(BASE_PATH, "out/attentions")

E_C = {
        'train': os.path.join(BASE_PATH, "datasets/E-c/E-c-En-train.txt"),
        'dev': os.path.join(BASE_PATH, "datasets/E-c/E-c-En-dev.txt"),
        'gold': os.path.join(BASE_PATH, "datasets/E-c/E-c-En-test-gold.txt")
      }


# model settings
MODEL_EC = {
    "name": "EmotionClassification",
    "token_type": "word",
    "batch_train": 32,
    "batch_eval": 32,
    "epochs": 50,
    "embeddings_file": "ntua_twitter_affect_310",
    "embed_dim": 310,
    "embed_finetune": False,
    "embed_noise": 0.2,
    "embed_dropout": 0.1,
    "encoder_dropout": 0.3,
    "encoder_size": 250,
    "encoder_layers": 2,
    "encoder_bidirectional": True,
    "attention": True,
    "attention_layers": 2,
    "attention_activation": "tanh",
    "attention_dropout": 0.2,
    "base": 0.56,
    "patience": 20,
    "weight_decay": 0.0,
    "clip_norm": 1,
}
