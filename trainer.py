"""
This file contains functions with logic that is used in almost all models,
with the goal of avoiding boilerplate code (and bugs due to copy-paste),
such as training pipelines.

"""

import os
import torch
import numpy

from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, jaccard_similarity_score

from lib.utils import load_word_vectors, load_datasets
from lib.config import DEVICE, BASE_PATH
from lib.training import Checkpoint, EarlyStop, Trainer
from lib.nn.models import ModelWrapper


def load_embeddings(model_conf):
    word_vector_file = os.path.join(BASE_PATH, "embeddings", "{}.txt".format(model_conf["embeddings_file"]))
    word_vectors_size = model_conf["embed_dim"]
    print("Loading word embeddings: {} ...".format(model_conf["embeddings_file"]))

    return load_word_vectors(word_vector_file, word_vectors_size)


def get_pipeline(task, loss_func=None, eval=False):
    """
    Generic classification pipeline
    Args:
        task: available tasks
            - "clf": multiclass classification
            - "bclf": binary classification
            - "mclf": multilabel classification
            - "reg": regression
        loss_func: the loss function
        eval: set to True if the pipeline will be used
                for evaluation and not for training.
                Note: this has nothing to do with the mode
                of the model (eval or train). If the pipeline will be used
                for making predictions, then set to True.

    """
    def pipeline(nn_model, curr_batch):
        # get the inputs (batch)
        inputs, labels, lengths, indices = curr_batch

        if task in ["mclf"]:
            labels = labels.float()

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        lengths = lengths.to(DEVICE)

        outputs, attentions = nn_model(inputs, lengths)

        if eval:
            return outputs, labels, attentions, None

        if task == "mclf":
            loss = loss_func(outputs.squeeze(), labels) # TODO

        return outputs, labels, attentions, loss

    return pipeline


def get_metrics(task):
    _metrics = {
        "mclf": {
            "accuracy": lambda y, y_hat: jaccard_similarity_score(
                                                    numpy.array(y), 
                                                    numpy.array(y_hat)),
            "f1-macro": lambda y, y_hat: f1_score(numpy.array(y),
                                                  numpy.array(y_hat),
                                                  average='macro'),
            "f1-micro": lambda y, y_hat: f1_score(numpy.array(y),
                                                  numpy.array(y_hat),
                                                  average='micro'),
        }
    }
    _monitor = {"mclf": "accuracy"}
    _mode = {"mclf": "max"}

    metrics = _metrics[task]
    monitor = _monitor[task]
    mode = _mode[task]

    return metrics, monitor, mode


def define_trainer(task,
                   config,
                   name,
                   datasets,
                   monitor,
                   finetune=None):
    """
    Args:
        task: available tasks
            - "mclf": multilabel classification
        config: model configurations
        name: task name
        datasets: datasets
        monitor: data as validation
        finetune: whether to fine tune parameters
        disable_cache: whether to use cache

    Returns:
        trainer

    """

    _config = config

    # Loading word embeddings
    word2idx = None
    if _config["token_type"] == "word":
        word2idx, idx2word, embeddings = load_embeddings(_config)

    # Constructing the pytorch datasets
    '''
    datasets = {
        "train": (X_train, y_train),
        "dev": (X_dev, y_dev),
        "test": (X_test, y_test),
    }    
    '''
    loaders = load_datasets(datasets,                               # DataLoader
                            train_batch_size=_config["batch_train"],
                            eval_batch_size=_config["batch_eval"],
                            token_type=_config["token_type"],
                            params=name,
                            word2idx=word2idx)

    # Defining the model which will be trained and its parameters
    out_size = 1
    if task == "mclf":
        out_size = len(loaders["train"].dataset.labels[0])

    num_embeddings = None

    model = ModelWrapper(embeddings=embeddings,
                         out_size=out_size,
                         num_embeddings=num_embeddings,
                         finetune=finetune,
                         **_config)
    model.to(DEVICE)
    print(model)

    # Loss function and optimizer
    if task == "mclf":
        loss_func = torch.nn.MultiLabelSoftMarginLoss() # same as sigmoid + BCE
        # loss_func = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Invalid task!")

    # p.requires_grad == True, X
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # print("Optimize parameters:")
    # print(list(parameters))
    # weight_decay: L2 penalty
    optimizer = torch.optim.Adam(parameters,
                                 weight_decay=config["weight_decay"])

    # Trainer
    if task == "mclf":
        pipeline = get_pipeline("mclf", loss_func)

    metrics, monitor_metric, mode = get_metrics(task)

    checkpoint = Checkpoint(name=name, model=model, model_conf=config,
                            monitor=monitor, keep_best=True, scorestamp=True,
                            timestamp=True, metric=monitor_metric, mode=mode,
                            base=config["base"])
    
    early_stopping = EarlyStop(metric=monitor_metric, mode=mode,
                               monitor=monitor,
                               patience=config["patience"])

    trainer = Trainer(model=model,
                      loaders=loaders,
                      task=task,
                      config=config,
                      optimizer=optimizer,
                      pipeline=pipeline,
                      metrics=metrics,
                      use_exp=True,
                      checkpoint=checkpoint,
                      early_stopping=early_stopping)

    return trainer

def model_training(trainer, epochs, checkpoint=False):
    print("Training...")
    for epoch in range(epochs):
        trainer.train()
        trainer.eval()

        '''
        if unfreeze > 0:
            if epoch == unfreeze:
                print("Unfreeze transfer-learning model...")
                subnetwork = trainer.model.feature_extractor
                if isinstance(subnetwork, ModuleList):
                    for fe in subnetwork:
                        unfreeze_module(fe.encoder, trainer.optimizer)
                        unfreeze_module(fe.attention, trainer.optimizer)
                else:
                    unfreeze_module(subnetwork.encoder, trainer.optimizer)
                    unfreeze_module(subnetwork.attention, trainer.optimizer)    
        '''
        print()

        if checkpoint:
            trainer.checkpoint.check()

        if trainer.early_stopping.stop():
            print("Early stopping...")
            break
