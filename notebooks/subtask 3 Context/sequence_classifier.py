import logging
import os
import sys

from datasets import load_metric

import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
import torch
from torch.utils.data.dataset import Dataset
import argparse
import os
from pathlib import Path
from torch.optim import AdamW
import pytorch_lightning as pl
from torchmetrics import Accuracy
from datetime import datetime 
from pathlib import Path
from pytorch_lightning import loggers as pl_loggers
import time
from argparse import Namespace
import json
import shutil
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
logger = logging.getLogger(__name__)

LABEL_TO_INT = {"action":{"Start": 1, "Stop": 2, "Increase":3, "Decrease": 4,
                          "UniqueDose": 5, "OtherChange":6, "Unknown": 0},
                "event_input": {"Disposition": 0, "NoDisposition": 1,
                          "Undetermined":2},
                "actor": {"Physician": 0, "Patient": 1, "Unknown":2},
                "certainty": {"Certain": 1, "Hypothetical": 2, "Conditional":3,
                              "Unknown": 0},
                "negation":  {"Negated": 1, "NotNegated": 0},
                "temporality": {"Past": 1, "Present": 2, "Future":3, 
                                "Unknown": 0}}
INT_TO_LABEL = {"action": {0: "O", 1: "B-STA", 2: "B-STO", 3: "B-INC", 4: "B-DEC",
                           5:"B-UNI", 6:"B-OTH"},
                "event_input":{0:"B-DIS", 1:"B-NOD", 2:"O"},
                "actor":  {0:"B-PHY", 1:"B-PAT", 2:"O"},
                "certainty":{0: "O", 1: "B-CER", 2: "B-HYP", 3: "B-CON"},
                "negation":{0: "O", 1: "B-NEG"},
                "temporality": {0: "O", 1: "B-PAS", 2: "B-PRE", 3: "B-FUT"}}
    

class BaseModel(pl.LightningModule):
    def __init__(
        self,
        **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        logger.info("Initilazing BaseModel")
        super().__init__()
        self.save_hyperparameters() #save hyperparameters to checkpoint
        self.step_count = 0
        self.output_dir = Path(self.hparams.output_dir)
        self.model = self._load_model()
        self.metric = load_metric("seqeval")
        self.accuracy = Accuracy()

    def _load_model(self):
        raise NotImplementedError

    def forward(self, **inputs):
        return self.model(**inputs)

    def batch2input(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        input = self.batch2input(batch)
        labels = input['labels']
        loss, pred_labels, _ = self(**input)
        true_predictions = [[INT_TO_LABEL[self.hparams.task][prediction]] 
                            for prediction in pred_labels.tolist()]
        true_labels = [[INT_TO_LABEL[self.hparams.task][label]] 
                       for label in labels.view(-1).tolist()]
        results = self.metric.compute(predictions=true_predictions, 
                                      references=true_labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1', results["overall_f1"], prog_bar=True)
        self.log('train_acc',results["overall_accuracy"], prog_bar=True)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input = self.batch2input(batch)
        labels = input['labels']
        loss, pred_labels, _ = self(**input)
        true_predictions = [[INT_TO_LABEL[self.hparams.task][prediction]] 
                            for prediction in pred_labels.tolist()]
        true_labels = [[INT_TO_LABEL[self.hparams.task][label]] 
                       for label in labels.view(-1).tolist()]
        results = self.metric.compute(predictions=true_predictions, 
                                      references=true_labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1', results["overall_f1"], prog_bar=True)
        self.log('val_acc',results["overall_accuracy"], prog_bar=True)

    def test_step(self, batch, batch_nb):
        input = self.batch2input(batch)
        labels = input['labels']
        loss, pred_labels, _ = self(**input)
        true_predictions = [[INT_TO_LABEL[self.hparams.task][prediction]] 
                            for prediction in pred_labels.tolist()]
        true_labels = [[INT_TO_LABEL[self.hparams.task][label]] 
                       for label in labels.view(-1).tolist()]
        results = self.metric.compute(predictions=true_predictions, 
                                      references=true_labels
                                      )
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_f1', results["overall_f1"], prog_bar=True)
        self.log('test_acc',results["overall_accuracy"], prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        # optimizer = SGD(model.parameters(), lr=self.hparams.learning_rate)
        optimizer = AdamW(model.parameters(), lr=self.hparams.learning_rate)

        self.opt = optimizer
        return [optimizer]

    def setup(self, stage):
        if stage == "fit":
            self.train_loader = self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.get_dataloader("dev", self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("dev", self.hparams.eval_batch_size, shuffle=False) # change back to test once we have test data

    @staticmethod
    def add_generic_args(parser, root_dir) -> None:
        parser.add_argument(
            "--max_epochs",
            default=10,
            type=int,
            help="The number of epochs to train your model.",
        )
        ############################################################
        ## WARNING: set --gpus 0 if you do not have access to GPUS #
        ############################################################
        parser.add_argument(
            "--gpus",
            default=1,
            type=int,
            help="The number of GPUs allocated for this, it is by default 1. Set to 0 for no GPU.",
        )
        parser.add_argument(
            "--output_dir",
            default=None,
            type=str,
            required=True,
            help="The output directory where the model predictions and checkpoints will be written.",
        )
        parser.add_argument("--do_train", action="store_true", default=True, help="Whether to run training.")
        parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
        parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
        parser.add_argument(
            "--data_dir",
            default="./",
            type=str,
            help="The input data dir. Should contain the training files.",
        )
        parser.add_argument("--learning_rate", default=1e-2, type=float, help="The initial learning rate for training.")
        parser.add_argument("--num_workers", default=16, type=int, help="kwarg passed to DataLoader")
        parser.add_argument("--num_train_epochs", dest="max_epochs", default=3, type=int)
        parser.add_argument("--train_batch_size", default=16, type=int)
        parser.add_argument("--eval_batch_size", default=16, type=int)
        parser.add_argument("--task", default="task2", type=str)


def generic_train(
    model: BaseModel,
    args: argparse.Namespace,
    early_stopping_callback=False,
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    **extra_train_kwargs
):
    # init model
    odir = Path(model.hparams.output_dir)
    odir.mkdir(exist_ok=True)
    log_dir = Path(os.path.join(model.hparams.output_dir, 'logs'))
    log_dir.mkdir(exist_ok=True)

    # Tensorboard logger
    pl_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_dir,
        version="version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
        name="",
        default_hp_metric=True
    )

    # add custom checkpoints
    ckpt_path = os.path.join(
        args.output_dir, pl_logger.version, "checkpoints",
    )
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_path, filename="{epoch}-{val_acc:.2f}", monitor="val_acc", mode="max", save_top_k=1, verbose=True
        )

    train_params = {}

    train_params["max_epochs"] = args.max_epochs

    if args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer.from_argparse_args(
        args,
        enable_model_summary=False,
        callbacks= [checkpoint_callback] + extra_callbacks,
        logger=pl_logger,
        **train_params,
    )

    if args.do_train:
        trainer.fit(model)
        # track model performance under differnt hparams settings in "Hparams" of TensorBoard
        pl_logger.log_hyperparams(params=model.hparams, metrics={'hp_metric': checkpoint_callback.best_model_score.item()})
        pl_logger.save()

        # save best model to `best_model.ckpt`
        target_path = os.path.join(ckpt_path, 'best_model.ckpt')
        logger.info(f"Copy best model from {checkpoint_callback.best_model_path} to {target_path}.")
        shutil.copy(checkpoint_callback.best_model_path, target_path)

    
    # Optionally, predict on test set and write to output_dir
    if args.do_predict:
        best_model_path = os.path.join(ckpt_path, "best_model.ckpt")
        model = model.load_from_checkpoint(best_model_path)
        return trainer.test(model)
    
    return trainer


class BERTSST2Dataset(Dataset):
    """
    Using dataset to process input text on-the-fly
    """
    def __init__(self, tokenizer, data):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = 512 # assigned based on length analysis of training set

    def __getitem__(self, index):
        note = []
        label, text = int(self.data[index][0]), self.data[index][1]
        return text, label

    def collate_fn(self, batch_data):
        texts, labels = list(zip(*batch_data))
        encodings = self.tokenizer(list(texts), padding=True, truncation=True, 
                                   max_length=self.max_len, return_tensors= 'pt')
        return (
                encodings['input_ids'],
                encodings['attention_mask'],
                torch.LongTensor(labels).view(-1,1)
               )

    def __len__(self):
        return len(self.data)

class BERT_PL(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        
    def _load_model(self):
        num_labels = len(INT_TO_LABEL[self.hparams.task])
        model_config = AutoConfig.from_pretrained(
            self.hparams.model_name,
            num_labels=num_labels,
        )
        return AutoModelForSequenceClassification.from_pretrained(self.hparams.model_name, config=model_config)

    def forward(self, **args):
        outputs = self.model(**args)
        loss, logits = outputs[0], outputs[1]
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, predicted_labels, []

    def get_dataloader(self, type_path, batch_size, shuffle=False):
        # todo add dataset path
        datapath = os.path.join(self.hparams.data_dir, 
                                f"{self.hparams.task}_{type_path}.json")
        with open(datapath) as json_file:
            data = [json.loads(line) for line in json_file]
        parsed_data =[]
        for line in data:
            parsed_data.append([LABEL_TO_INT[self.hparams.task_bert][line["label"]], line["text"]])
        dataset = BERTSST2Dataset(self.tokenizer, parsed_data)

        logger.info(f"Loading {type_path} data and labels from {datapath}")
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            collate_fn=dataset.collate_fn
        )
        
        return data_loader    

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        optimizer = AdamW(model.parameters(), 
                          lr=self.hparams.learning_rate,
                          eps=1e-8)
        self.opt = optimizer
        return [optimizer]
    
    def batch2input(self, batch):
        return {"input_ids": batch[0], "labels": batch[2], "attention_mask": batch[1]}

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--model_name",
            default=None,
            type=str,
            required=True,
            help="Pretrained tokenizer name or path",
        )
        parser.add_argument(
            "--optimizer",
            default="adamw",
            type=str,
            required=True,
            help="Whether to use SGD or not",
        )
        parser.add_argument(
            "--task_bert",
            default="task2",
            type=str,
            required=True,
            help="Which task this classification is for",
        )
        return parser


def classify(data_dir, task, optimizer="adamw", max_epochs=5, learning_rate=0.00005,
             eval_batch_size=16, num_workers=16, train_batch_size=16):
    print(os.path.dirname(os.path.realpath(__file__)))
    mock_args = f"--data_dir {data_dir} --output_dir data/models/{task} --optimizer {optimizer} \
    --model_name emilyalsentzer/Bio_ClinicalBERT --learning_rate {learning_rate} --max_epochs {max_epochs} \
    --eval_batch_size {eval_batch_size} --num_workers {num_workers} --do_predict \
    --train_batch_size {train_batch_size} --task {task} --task_bert {task}" # change model_name and file name (task) here
    # For example, if the data is "train_event.json/dev_event.json", change task to event

    # load hyperparameters
    parser = argparse.ArgumentParser()
    BaseModel.add_generic_args(parser, os.getcwd())
    parser = BERT_PL.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args(mock_args.split())
    print(args)
    # fix random seed to make sure the result is reproducible
    pl.seed_everything(args.seed)

    # If output_dir not provided, a folder will be generated in pwd
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "./results",
            f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(args.output_dir)
    dict_args = vars(args)
    model = BERT_PL(**dict_args)
    trainer = generic_train(model, args)

