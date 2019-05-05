import os
import json
import math
import time
import sys
import numpy
import pickle
# import collections

import torch
from torch.nn.utils import clip_grad_norm_

from lib.config import MODEL_PATH, EXPS_PATH
from lib.visdom_tools import Experiment, Metric


def epoch_progress(loss, epoch, batch, batch_size, dataset_size):
    # print(batch)    # idx_batch 6839/32
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Batch Loss ({}): {:.4f}'.format(epoch, batch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()


def _get_predictions(posteriors, task):
    """
    Args:
        posteriors (numpy.array):

    """

    if task == "mclf":
        # sign: < 0 => -1, > 0 => 1, = 0 => 0
        # clip: < a_min => = a_min
        predicted = numpy.clip(numpy.sign(posteriors), a_min=0,
                               a_max=None)
    else:
        raise ValueError

    return predicted


def predict(model, pipeline, dataloader, task, mode="eval"):
    """
    Pass a dataset(dataloader) to the model and get the predictions
    Args:
        dataloader (DataLoader): a torch DataLoader which will be used for
            evaluating the performance of the model
        mode: set the operation mode of the model.
            - "eval" : disable regularization layers
            - "train" : enable regularization layers (MC eval)
    """
    if mode == "eval":
        model.eval()
    elif mode == "train":
        model.train()
    else:
        raise ValueError

    posteriors = []
    y_pred = []
    y = []
    attentions = []
    total_loss = 0

    # flag = 0

    for i_batch, sample_batched in enumerate(dataloader, 1):    # enumerate(sequence, [start=0])
        outputs, labels, atts, loss = pipeline(model, sample_batched)

        if loss is not None:
            total_loss += loss.item()

        # get the model posteriors
        posts = outputs.data.cpu().numpy()

        # get the actual predictions (classes and so on...)
        '''
        if flag == 0:
            print("post shape") # 2
            print(len(posts.shape))
            flag = 1
        if len(posts.shape) == 1:
            predicted = _get_predictions(numpy.expand_dims(posts, axis=0),
                                         task)
        else:
            predicted = _get_predictions(posts, task)
        
        '''

        predicted = _get_predictions(posts, task)

        # to numpy
        labels = labels.data.cpu().numpy().squeeze().tolist()
        predicted = predicted.squeeze().tolist()
        posts = posts.squeeze().tolist()
        if atts is not None:
            atts = atts.data.cpu().numpy().squeeze().tolist()

        '''
        if not isinstance(labels, collections.Iterable):
            print('not isinstance(labels, collections.Iterable)')
            labels = [labels]
            predicted = [predicted]
            posts = [posts]
            if atts is not None:
                atts = [atts]        
        '''

        '''
        # make transformations to the predictions
        label_transformer = dataloader.dataset.label_transformer
        # print(label_transformer)
        if label_transformer is not None:
            labels = [label_transformer.inverse(x) for x in labels]
            labels = numpy.array(labels)
            predicted = [label_transformer.inverse(x) for x in predicted]
            predicted = numpy.array(predicted)
        '''

        y.extend(labels)
        y_pred.extend(predicted)
        posteriors.extend(posts)
        if atts is not None:
            attentions.extend(atts)

    avg_loss = total_loss / i_batch

    return avg_loss, (y, y_pred), posteriors, attentions


class MetricWatcher:
    """
    Base class which monitors a given metric on a Trainer object
    and check whether the model has been improved according to this metric
    """

    def __init__(self, metric, monitor, mode="max", base=None):
        self.best = base
        self.metric = metric
        self.mode = mode
        self.monitor = monitor
        self.scores = None  # will be filled by the Trainer instance

    def has_improved(self):

        # get the latest value for the desired metric
        value = self.scores[self.metric][self.monitor][-1]

        # init best value
        if self.best is None or math.isnan(self.best):
            self.best = value
            return True

        if(
            self.mode == "min" and value < self.best
            or
            self.mode == "max" and value > self.best
        ):  
            self.best = value   # the performance of the model has been improved :)
            return True
        else:
            # no improvement :(
            return False


class EarlyStop(MetricWatcher):
    def __init__(self, metric, monitor, mode="min", patience=0):
        """
        Args:
            patience (int): for how many epochs to wait, for the performance
                to improve.
            mode (str, optional): Possible values {"min","max"}.
                - "min": save the model if the monitored metric is decreased.
                - "max": save the model if the monitored metric is increased.

        """
        MetricWatcher.__init__(self, metric, monitor, mode)
        self.patience = patience
        self.patience_left = patience
        self.best = None

    def stop(self):
        """
        Check whether we should stop the training

        """
        if self.has_improved():
            self.patience_left = self.patience  # reset patience
        else:
            self.patience_left -= 1  # decrease patience

        print("patience left:{}, best({})".format(self.patience_left, self.best))

        # if no more patience left, then stop training
        return self.patience_left < 0


class Checkpoint(MetricWatcher):
    def __init__(self, name, model, monitor, metric, model_conf, mode="min",
                 dir=None,
                 base=None,
                 timestamp=False,
                 scorestamp=False,
                 keep_best=False):
        """
        Args:
            model (nn.Module):
            name (str): the name of the model
            mode (str, optional): Possible values {"min","max"}.
                - "min": save the model if the monitored metric is decreased.
                - "max": save the model if the monitored metric is increased.
            dir (str): the directory in which the checkpoint files will be saved
            timestamp (bool): if True add a timestamp to the checkpoint files
            scorestamp (bool): if True add the score to the checkpoint files
            keep_best (bool): if True then keep only the best checkpoint

        """
        MetricWatcher.__init__(self, metric, monitor, mode, base)

        self.name = name
        self.dir = dir
        self.model = model
        self.model_conf = model_conf
        self.timestamp = timestamp
        self.scorestamp = scorestamp
        self.keep_best = keep_best
        self.last_saved = None

        if self.dir is None:
            self.dir = MODEL_PATH

    def _define_cp_name(self):
        """
        Define the checkpoint name
        
        """
        fname = [self.name]

        if self.scorestamp:
            score_str = "{:.4f}".format(self.best)
            fname.append(score_str)

        if self.timestamp:
            date_str = time.strftime("%Y-%m-%d_%H:%M")
            fname.append(date_str)

        return "_".join(fname)

    def _save_checkpoint(self):
        """
        A checkpoint saves:
            - the model itself
            - the model's config, which is required for loading related data,
            such the word embeddings, on which it was trained

        """
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        name = self._define_cp_name()
        file_cp = os.path.join(self.dir, name + ".model")
        file_conf = os.path.join(self.dir, name + ".conf")

        # remove previous checkpoint files, if keep_best is True
        if self.keep_best and self.last_saved is not None:
            os.remove(self.last_saved["model"])
            os.remove(self.last_saved["config"])

        # update last saved checkpoint files
        self.last_saved = {
            "model": file_cp,
            "config": file_conf
        }

        # save the checkpoint files (model, model config)
        torch.save(self.model, file_cp)
        with open(file_conf, 'wb') as f:
            pickle.dump(self.model_conf, f)

    def check(self):
        """
        Check whether the model has improved and if so, then save a checkpoint

        """
        if self.has_improved():
            print("Improved model ({}:{:.4f})! "
                  "Saving checkpoint...".format(self.metric, self.best))
            self._save_checkpoint()


class Trainer:
    def __init__(self, model,
                 loaders,
                 optimizer,
                 pipeline,
                 config,
                 task="mclf",
                 use_exp=False,
                 metrics=None,
                 eval_train=True,
                 checkpoint=None,
                 early_stopping=None):
        """
         The Trainer is responsible for training a model.
         It holds a set of variables that helps us to abstract
         the training process.

        Args:
            use_exp (bool): if True, use the integrated experiment
                manager. In order to utilize the visualizations provided
                by the experiment manager you should:
                    - run `python -m visdom.server` in a terminal.
                    - access visdom by going to http://localhost:8097

                    https://github.com/facebookresearch/visdom#usage

            model (nn.Module): the pytorch model
            optimizer (torch.optim.Adam)
            pipeline (callable): a callback function, which defines the training
                pipeline. it must return 3 things (outputs, labels, loss):
                    - outputs: the outputs (predictions) of the model
                    - labels: the gold labels
                    - loss: the loss

            config (): the config instance with the hyperparams of the model
            task (string): you can choose between {"clf", "reg", "mclf"},
                for classification and regression respectively.
            metrics (dict): a dictionary with the metrics that will be used
                for evaluating the performance of the model.
                - key: string with the name of the metric.
                - value: a callable, with arguments (y, y_hat) tha returns a score.
            eval_train (bool): if True, the at the end of each epoch evaluate
                the performance of the model on the training dataset.
            early_stopping (EarlyStop)
            checkpoint (Checkpoint)

        """
        self.use_exp = use_exp
        self.model = model
        self.task = task
        self.config = config
        self.eval_train = eval_train
        self.loaders = loaders
        self.optimizer = optimizer
        self.pipeline = pipeline
        self.checkpoint = checkpoint
        self.early_stopping = early_stopping
        self.metrics = {} if metrics is None else metrics
        self.running_loss = 0.0
        self.epoch = 0

        # Initializations
        _dataset_names = list(self.loaders.keys())
        _metric_names = list(self.metrics.keys()) + ["loss"]

        # init watched metrics
        self.scores = {metric: {d: [] for d in _dataset_names}
                       for metric in _metric_names}

        if self.checkpoint is not None:
            self.checkpoint.scores = self.scores
        if self.early_stopping is not None:
            self.early_stopping.scores = self.scores

        # init Experiment
        if use_exp:
            self.experiment = Experiment(name=self.config["name"],
                                         desc=str(self.model),
                                         hparams=self.config)

            for metric in _metric_names:
                self.experiment.add_metric(Metric(name=metric,
                                                  tags=_dataset_names,
                                                  vis_type="line"))

    def __on_after_train(self):
        pass

    def __on_after_eval(self):
        # 4 - update the corresponding values in the experiment
        if self.use_exp:
            for score_name, score in self.scores.items():
                for tag, value in score.items():
                    self.experiment.metrics[score_name].append(tag, value[-1])

            self.experiment.update_plots()

    def train_loader(self, loader):
        """
        Run a pass of the model on a given dataloader

        """
        # switch to train mode -> enable regularization layers, such as Dropout
        self.model.train()

        running_loss = 0.0
        for i_batch, sample_batched in enumerate(loader, 1):
            # print(i_batch)
            # 1 - zero the gradients
            self.optimizer.zero_grad()
            # 2 - compute loss using the provided pipeline
            # print(type(sample_batched))
            # print(numpy.shape(sample_batched))
            outputs, labels, attentions, loss = self.pipeline(self.model,
                                                              sample_batched)
            # 3 - backward pass
            loss.backward()

            # just to be sure... clip gradients with norm > N.
            # apply it only if the model has an RNN in it.
            if len([m for m in self.model.modules()
                    if hasattr(m, 'bidirectional')]) > 0:
                clip_grad_norm_(self.model.parameters(),
                                self.config["clip_norm"])

            # 4 - update weights
            self.optimizer.step()

            running_loss += loss.item()

            # print statistics
            epoch_progress(loss=loss.item(),
                           epoch=self.epoch,
                           batch=i_batch,
                           batch_size=loader.batch_size,
                           dataset_size=len(loader.dataset))
        return running_loss

    def train(self):
        """
        Train the model for one epoch (on one or more dataloaders)

        """
        self.epoch += 1
        self.train_loader(self.loaders["train"])
        print()
        self.__on_after_train()

    def eval_loader(self, loader):
        """
        Evaluate a dataloader and update the corresponding scores and metrics

        """
        # 1 - evaluate the dataloader
        avg_loss, (y, y_pred), posteriors, attentions = predict(
            self.model,
            self.pipeline,
            loader,
            self.task,
            "eval")

        return avg_loss, (y, y_pred), posteriors, attentions

    def eval(self):
        """
        Evaluate the model on each dataset and update the corresponding metrics.
        The function is normally called at the end of each epoch.

        """

        for k, v in self.loaders.items():
            avg_loss, (y, y_pred), _, _ = self.eval_loader(v)

            scores = self.__calc_scores(y, y_pred)

            self.__log_scores(scores, avg_loss, k)

            scores["loss"] = avg_loss

            for name, value in scores.items():
                self.scores[name][k].append(value)

        self.__on_after_eval()

    def __calc_scores(self, y, y_pred):
        return {name: metric(y, y_pred)
                for name, metric in self.metrics.items()}

    def __log_scores(self, scores, loss, tag):
        """
        Log the scores of a dataset (tag) on the console
        Args:
            scores: a dictionary of (metric_name, value)
            loss: the loss of the model on an epoch
            tag: the dataset (name)

        """
        print("\t{:6s} - ".format(tag), end=" ")
        for name, value in scores.items():
            print(name, '{:.4f}'.format(value), end=", ")
        print(" Loss:{:.4f}".format(loss))

    def log_training(self, name, desc):

        results = {}
        scores = {k: v for k, v in self.scores.items() if k != "loss"}

        results["name"] = name
        results["desc"] = desc
        results["scores"] = scores

        path = os.path.join(EXPS_PATH, self.config["name"])

        # JSON
        json_file = path + ".json"
        try:
            with open(json_file) as f:
                data = json.load(f)
        except:
            data = []

        # adding experiment results
        data.append(results)

        with open(json_file, 'w') as f:
            json.dump(data, f)
