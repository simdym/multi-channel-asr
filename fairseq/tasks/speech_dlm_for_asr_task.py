import logging
import os
import glob
from dataclasses import dataclass, field
from typing import Optional
from collections import OrderedDict

import numpy as np
import torch
from fairseq import utils

from fairseq.data import (
    SpeechDLMForASRDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    data_utils,
    encoders,
    indexed_dataset,
    Dictionary
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, LegacyFairseqTask, register_task
from omegaconf import II

from typing import List

@dataclass
class SpeechDLMForASRTaskConfig(FairseqDataclass):
    data_path: str = field(
        default=".", metadata={"help": "path to data directory"}
    )
    input_dict_path: str = field(
        default=".", metadata={"help": "path to input dictionary"}
    )
    output_dict_path: str = field(
        default=".", metadata={"help": "path to output dictionary"}
    )
    channels: str = field(
        default="0,1", metadata={"help": "channel names to use for the model"}
    )
    prepend_bos_token: Optional[bool] = field(
        default=False, metadata={"help": "prepend beginning of sentence token (<s>)"}
    ) # Remove this?
    prepend_token: Optional[str] = field(
        default="eos", metadata={"help": "prepend token to source sequences"}
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    tokens_per_sample: int = field(
        default=1024, metadata={"help": "max number of tokens per sample for LM dataset"}
    )
    shuffle: bool = field(
        default=False, metadata={"help": "shuffle dataset"}
    )




@register_task("speech_dlm_for_asr_task", dataclass=SpeechDLMForASRTaskConfig)
class SpeechDLMForASRTask(FairseqTask):
    """Task for using the SpeechDLM model from https://arxiv.org/pdf/2203.16502.pdf
    for automatic speech recognition:
    
    It create a multi-channel dataset (SpeechDLMDataset) from multiple
    dictionaries.

    Args:
        input_dicts (~fairseq.data.Dictionary): the dictionaries for
            each input channel of the SpeechDLM model
        output_dicts (~fairseq.data.Dictionary): the dictionaries
            for the output of each channel of the SpeechDLM model.
    """

    def __init__(self, args, input_dict, output_dict):
        super().__init__(args)
        self.args = self.cfg # Rename cfg to args (args is used in the rest of the code)
        self.input_dict = input_dict
        self.output_dict = output_dict

        if args.channels is not None:
            self.channels = sorted(args.channels.split(","))
        else:
            self.channels = ["0", "1"]

    @classmethod
    def setup_dictionary(cls, args, **kwargs):
        """The dictionaries will be a dict over channel keys and values of type
        ~fairseq.data.Dictionary.
        """
        input_dict = Dictionary.load(args.input_dict_path)
        output_dict = Dictionary.load(args.output_dict_path)
        output_dict.add_symbol("_") # Add the CTC blank symbol

        return (
            input_dict,
            output_dict
        )
    
    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        input_dict, output_dict = cls.setup_dictionary(args, **kwargs)
        return cls(args, input_dict, output_dict)
    
    def load_dataset(
            self, split: str, epoch=1, combine=False, **kwargs
        ):
        paths = utils.split_paths(self.args.data_path)
        assert len(paths) > 0

        data_path = paths[(epoch - 1) % len(paths)] # And this??
        split_path = os.path.join(data_path, split)
        
        self.datasets[split] = SpeechDLMForASRDataset.from_jsons(
            args=self.args,
            data_path=split_path,
            channels=self.channels,
            src_dict=self.input_dict,
            tgt_dict=self.output_dict,
            prepend_token=self.args.prepend_token,
        )