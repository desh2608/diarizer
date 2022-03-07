#!/usr/local/env python
# coding: utf-8
# Copyright 2022  Johns Hopkins University (author: Desh Raj)

import argparse
from pathlib import Path
from copy import deepcopy
import logging

import pytorch_lightning as pl
from pyannote.database import get_protocol
from pyannote.audio import Model
from pyannote.audio.tasks import Segmentation


def read_args():
    parser = argparse.ArgumentParser(description="Fine-tune Pyannote model on data")
    parser.add_argument(
        "dataset",
        type=str,
        help="Name of dataset",
        choices=["AMI", "AISHELL-4", "AliMeeting"],
    )
    parser.add_argument("exp_dir", type=str, help="Experiment directory")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = read_args()

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    # Create dataset
    logging.info("Loading dataset")
    ami = get_protocol(f"{args.dataset}.SpeakerDiarization.only_words")

    # Get pretrained segmentation model
    logging.info("Loading pretrained model")
    pretrained = Model.from_pretrained("pyannote/segmentation")

    # Create new segmentation task for dataset
    logging.info("Creating new segmentation task")
    seg_task = Segmentation(ami, duration=5.0, max_num_speakers=4, num_workers=4)

    # Copy pretrained model and override task
    logging.info("Copying pretrained model")
    finetuned = deepcopy(pretrained)
    finetuned.task = seg_task

    # Create trainer
    logging.info("Creating trainer")
    trainer = pl.Trainer(gpus=1, max_epochs=1, default_root_dir=exp_dir)

    # Trainer model
    logging.info("Training model")
    trainer.fit(finetuned)
