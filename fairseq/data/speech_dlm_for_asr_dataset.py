from collections import OrderedDict

import numpy as np
import torch
import os
import glob
import json
import re
from tqdm import tqdm
from dataclasses import dataclass, field

from fairseq.data import FairseqDataset, LanguagePairDataset, data_utils, Dictionary
from fairseq.tokenizer import tokenize_line

from typing import Dict, List, Optional


class SpeechDLMForASRDataset(FairseqDataset):
    """The dataset used to train the SpeechDLM model from the paper:
    https://arxiv.org/pdf/2203.16502.pdf, but for ASR.

    The input datasets is expected to be a dict over channel names with the values
    being instances of :class:`~fairseq.data.LanguagePairDataset`.

    Each element of SpeechDLMDataset is a dictionary with the following keys:
        - `id` (int) : index of the item
        - `source` (OrderedDict[str, Tensor of shape (src_len,)]) : dictionary over
            channels with the values containing the input unit tokens
        - `target` (OrderedDict[str, Tensor of shape (tgt_len,)]) : dictionary over
            channels with the values containing the tgt character tokens
    
    Args:
        datasets (Dict[str, ~fairseq.data.LanguagePairDataset]): a dictionary over 
            channel with :class:`~fairseq.data.LanguagePairDataset` instances.
        shuffle (bool, optional): shuffle the elements before batching
            (default: True).
    """

    def __init__(
        self, datasets, shuffle=True
    ):
        if isinstance(datasets, dict):
            datasets = OrderedDict(datasets)
        assert isinstance(
            datasets, OrderedDict
        ), "datasets is expected to be an instance of Dictionary or OrderedDict"
        assert datasets, "datasets is None"
        for dataset in datasets.values():
            assert isinstance(
                dataset, LanguagePairDataset
            ), "Each value of datasets is expected to be an instance of MonolingualDataset"

        self.datasets = datasets
        self.shuffle = shuffle

        self.src_vocab = next(iter(datasets.values())).src_dict
        self.tgt_vocab = next(iter(datasets.values())).tgt_dict
        self.length = len(next(iter(datasets.values())))

        # Source lengths are the same for all channels
        self.src_sizes = next(iter(datasets.values())).src_sizes
        
        # Target lengths can be different for each channel
        self.tgt_sizes = []
        for i in range(len(next(iter(datasets.values())))):
            self.tgt_sizes.append(
                dict(
                    (key, dataset.tgt_sizes[i])
                    for (key, dataset) in self.datasets.items()
                )
            )

    def __getitem__(self, index):
        source = OrderedDict(
            [
                (key, dataset[index]["source"])
                for (key, dataset) in self.datasets.items()
            ]
        )

        target = OrderedDict(
            [
                (key, dataset[index]["target"])
                for (key, dataset) in self.datasets.items()
            ]
        )

        #TODO: Add support for prepend_token

        return {
            "id": index,
            "src_tokens": source,
            "tgt_tokens": target,
            "src_length": self.src_sizes[index],
            "tgt_length": self.tgt_sizes[index]
        }
    
    def __len__(self):
        return self.length
    
    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        """
        
        if len(samples) == 0:
            return {}

        def merge(key, dict=None, max_size=None):
            """Merge a list of items to form a mini-batch by padding to the length of the longest item."""
            pad_idx = dict.pad()
            eos_idx = dict.eos()

            if samples[0][key] is None:
                return None
            res = OrderedDict()
            for channel in samples[0][key]:
                data = [s[key][channel] for s in samples]
                res[channel] = data_utils.collate_tokens(
                    data,
                    pad_idx,
                    eos_idx,
                    left_pad=False,
                    move_eos_to_beginning=False
                )
            return res

        def merge_lengths(key):
            return {
                channel: torch.tensor(
                    [s[key][channel] for s in samples], dtype=torch.long
                )
                for channel in samples[0][key]
            }
        
        src_tokens_batch = merge("src_tokens", self.src_vocab)
        tgt_tokens_batch = merge("tgt_tokens", self.tgt_vocab)
        tgt_lengths_batch = merge_lengths("tgt_length")


        return {
            "id": torch.tensor([s["id"] for s in samples], dtype=torch.long),
            "nsentences": len(samples),
            "ntokens": sum(len(item) for s in samples for item in s["src_tokens"].values()),
            "net_input": {
                "src_tokens": src_tokens_batch,
                "src_lengths": torch.tensor(
                    [s["src_length"] for s in samples], dtype=torch.long
                ) # Source lengths are the same for both channels
            },
            "net_target": {
                "tgt_tokens": tgt_tokens_batch,
                "tgt_lengths": tgt_lengths_batch # Target lengths can be different for each channel
            }
        }


    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.src_sizes[index]
    
    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.src_sizes[index]
    
    @property
    def sizes(self):
        return self.src_sizes
    
    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)
    
    @staticmethod
    def from_jsons(
        args,
        data_path: str,
        channels: List[str],
        src_dict: Dictionary,
        tgt_dict: Dictionary,
        prepend_token: Optional[str]=None
    ):
        json_paths = glob.glob(os.path.join(data_path, "*.json"))

        lang_pair_datasets = {}
        for channel in channels:
            src_elements = []
            tgt_elements = []
            for id, json_path in enumerate(tqdm(json_paths, desc=f"Loading channel {channel}")):
                with open(json_path, "r") as f:
                    sample_dict = json.load(f)

                slice = sample_dict["file_id"] + "_" + sample_dict["slice_id"]
                # Skip sample if there is no units
                if sample_dict["units"] is None or channel not in sample_dict["units"]:
                    #print(f"Sample {slice} has no units")
                    continue
                
                # Skip sample if there is no text
                if sample_dict["text"] is None or channel not in sample_dict["text"]:
                    #print(f"Sample {slice} has no text")
                    continue

                src_len = len(sample_dict["units"][channel].split(" "))
                tgt_len = len(sample_dict["text"][channel])

                src_elements.append(
                    JsonElement(
                        id=id,
                        path=json_path,
                        field="units",
                        size=src_len,
                        channel=channel
                    )
                )

                tgt_elements.append(
                    JsonElement(
                        id=id,
                        path=json_path,
                        field="text",
                        size=tgt_len,
                        channel=channel
                    )
                )

            # Create JSON datasets
            src_dataset = JsonDataset(
                src_elements,
                src_dict,
                tokenizer_function=tokenize_line,
                replace_spaces_with=None,
                prepend_token=prepend_token
            )

            tgt_dataset = JsonDataset(
                tgt_elements,
                tgt_dict,
                tokenizer_function=tokenize_characters,
                replace_spaces_with="|",
                prepend_token=None
            )

            # Create wrapper LanguagePairDataset from JSON datasets
            lang_pair_datasets[channel] = LanguagePairDataset(
                src=src_dataset,
                src_sizes=src_dataset.sizes,
                src_dict=src_dict,
                tgt=tgt_dataset,
                tgt_sizes=tgt_dataset.sizes,
                tgt_dict=tgt_dict,
                shuffle=False,
                append_bos=False
            )

        return SpeechDLMForASRDataset(
            datasets=lang_pair_datasets,
            shuffle=args.shuffle
        )


SPACE_NORMALIZER = re.compile(r"\s+")

def tokenize_characters(characters):
    characters = SPACE_NORMALIZER.sub(" ", characters)
    characters = characters.strip()
    return list(characters)


@dataclass
class JsonElement:
    """Dataclass for elements in the JSON dataset."""
    id: int
    path: str # Path to the json file
    field: str # "units" or "text"
    size: int # Number of source tokens
    channel: str # Channel name


class JsonDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data: List[JsonElement],
            vocab_dict: Dictionary,
            tokenizer_function=tokenize_line,
            replace_spaces_with:Optional[str]=None,
            prepend_token: Optional[str]=None
        ):
        """Dataset for loading from JSON files.

        Args:
            data (List[JsonElement]): list of elements
            vocab_dict (Dictionary): the vocabulary dictionary
            tokenizer_function (Callable[[str], List[str]], optional): tokenizer function
                to use for the content. Defaults to tokenize_line.
            replace_spaces_with (Optional[str], optional): character to replace spaces with.
                Defaults to None.
        """
        self.data: List[JsonElement] = data
        self.vocab_dict = vocab_dict
        self.tokenizer_function = tokenizer_function
        self.replace_spaces_with = replace_spaces_with

        if prepend_token == "eos":
            self.prepend_token = vocab_dict.eos()
        elif prepend_token == "bos":
            self.prepend_token = vocab_dict.bos()
        elif prepend_token == "pad" or prepend_token == "unk":
            raise ValueError(f"Prepend token {prepend_token} is not allowed")
        else:
            self.prepend_token = prepend_token
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Extract the content from the json file
        item = self.data[index]
        element_dict = self.load_json(item.path)
        content = element_dict[item.field][item.channel]

        # Replace spaces if specified
        if self.replace_spaces_with is not None:
            content = content.replace(" ", self.replace_spaces_with)
        
        # Prepend token if specified
        if self.prepend_token is not None:
            content = torch.cat(content.new([self.prepend_token]), content) # (Cast prepend_token to same type as content tensor with new() method)

        # Tokenize the content        
        return self.vocab_dict.encode_line(
            content,
            line_tokenizer=self.tokenizer_function,
            add_if_not_exist=False
        )
    
    def load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    @property
    def sizes(self):
        return [d.size for d in self.data]
