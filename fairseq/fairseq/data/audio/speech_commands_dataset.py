"""
for speech command dataset
Adapted from https://github.com/HazyResearch/state-spaces/blob/main/src/dataloaders/sc.py
which is
adapted from https://github.com/dwromero/ckconv/blob/dc84dceb490cab2f2ddf609c380083367af21890/datasets/speech_commands.py
which is
adapted from https://github.com/patrick-kidger/NeuralCDE/blob/758d3a7134e3a691013e5cc6b7f68f277e9e6b69/experiments/datasets/speech_commands.py
"""


import os
import logging
import numpy as np
import torch
import torch.nn.functional as F

import pathlib
import tarfile
import urllib.request

import sklearn.model_selection
from .. import FairseqDataset, BaseWrapperDataset

logger = logging.getLogger(__name__)


def pad(channel, maxlen):
    channel = torch.tensor(channel)
    out = torch.full((maxlen,), channel[-1])
    out[: channel.size(0)] = channel
    return out


def subsample(X, y, subsample_rate):
    if subsample_rate != 1:
        X = X[:, ::subsample_rate, :]
    return X, y


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + ".pt")


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith(".pt"):
            tensor_name = filename.split(".")[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors


def normalise_data(X, y):
    train_X, _, _ = split_data(X, y)
    out = []
    for Xi, train_Xi in zip(X.unbind(dim=-1), train_X.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        mean = train_Xi_nonan.mean()  # compute statistics using only training data.
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    out = torch.stack(out, dim=-1)
    return out

def normalize_all_data(X_train, X_val, X_test):

    for i in range(X_train.shape[-1]):
        mean = X_train[:, :, i].mean()
        std = X_train[:, :, i].std()
        X_train[:, :, i] = (X_train[:, :, i] - mean) / (std + 1e-5)
        X_val[:, :, i] = (X_val[:, :, i] - mean) / (std + 1e-5)
        X_test[:, :, i] = (X_test[:, :, i] - mean) / (std + 1e-5)

    return X_train, X_val, X_test

def minmax_scale(tensor):
    min_val = torch.amin(tensor, dim=(1, 2), keepdim=True)
    max_val = torch.amax(tensor, dim=(1, 2), keepdim=True)
    return (tensor - min_val) / (max_val - min_val)

def mu_law_encode(audio, bits=8):
    """
    Perform mu-law companding transformation.
    """
    mu = torch.tensor(2**bits - 1)

    # Audio must be min-max scaled between -1 and 1
    audio = 2 * minmax_scale(audio) - 1

    # Perform mu-law companding transformation.
    numerator = torch.log1p(mu * torch.abs(audio))
    denominator = torch.log1p(mu)
    encoded = torch.sign(audio) * (numerator / denominator)

    # Quantize signal to the specified number of levels.
    return ((encoded + 1) / 2 * mu + 0.5).to(torch.int32)

def mu_law_decode(encoded, bits=8):
    """
    Perform inverse mu-law transformation.
    """
    mu = 2**bits - 1
    # Invert the quantization
    x = (encoded / mu) * 2 - 1

    # Invert the mu-law transformation
    x = torch.sign(x) * ((1 + mu)**(torch.abs(x)) - 1) / mu
    return x

def split_data(tensor, stratify):
    # 0.7/0.15/0.15 train/val/test split
    (
        train_tensor,
        testval_tensor,
        train_stratify,
        testval_stratify,
    ) = sklearn.model_selection.train_test_split(
        tensor,
        stratify,
        train_size=0.7,
        random_state=0,
        shuffle=True,
        stratify=stratify,
    )

    val_tensor, test_tensor = sklearn.model_selection.train_test_split(
        testval_tensor,
        train_size=0.5,
        random_state=1,
        shuffle=True,
        stratify=testval_stratify,
    )
    return train_tensor, val_tensor, test_tensor


class SpeechCommandsDataset(FairseqDataset):

    SUBSET_CLASSES = [
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    ]
    ALL_CLASSES = [
        "bed",
        "cat",
        "down",
        "five",
        "forward",
        "go",
        "house",
        "left",
        "marvin",
        "no",
        "on",
        "right",
        "sheila",
        "tree",
        "up",
        "visual",
        "yes",
        "backward",
        "bird",
        "dog",
        "eight",
        "follow",
        "four",
        "happy",
        "learn",
        "nine",
        "off",
        "one",
        "seven",
        "six",
        "stop",
        "three",
        "two",
        "wow",
        "zero",
    ]

    def __init__(
            self,
            partition: str,  # `train`, `val`, `test`
            length: int, # sequence length
            mfcc: bool,  # whether to use MFCC features (`True`) or raw features
            sr: int,  # subsampling rate: default should be 1 (no subsampling); keeps every kth sample
            dropped_rate: float,  # rate at which samples are dropped, lies in [0, 100.]
            path: str,
            all_classes: bool = False,
            gen: bool = False,  # whether we are doing speech generation
            discrete_input: bool = False,  # whether we are using discrete inputs
            resolution: int = 1,  # resolution of the input
    ):
        # compatible with fairseq
        if partition == 'valid':
            partition = 'val'

        self.dropped_rate = dropped_rate
        self.all_classes = all_classes
        self.gen = gen
        self.discrete_input = discrete_input
        self.resolution = resolution

        self.root = pathlib.Path(path)
        base_loc = self.root / "processed_data"

        if mfcc:
            data_loc = base_loc / "mfcc"
        elif gen:
            data_loc = base_loc / "gen"
        else:
            data_loc = base_loc / "raw"

            if self.dropped_rate != 0:
                data_loc = pathlib.Path(
                    str(data_loc) + "_dropped{}".format(self.dropped_rate)
                )

        if self.all_classes:
            data_loc = pathlib.Path(str(data_loc) + "_all_classes")

        if self.discrete_input:
            data_loc = pathlib.Path(str(data_loc) + "_discrete")

        if not os.path.exists(data_loc):
            if not os.path.exists(base_loc):
                os.makedirs(base_loc, exist_ok=True)
            
            # Load the already processed data files directly if available
            if os.path.exists(self.root / "train_X.pt"):
                logger.info(f"Loading pre-processed data from {self.root}")
                train_X = torch.load(self.root / "train_X.pt")
                train_y = torch.load(self.root / "train_y.pt")
                val_X = torch.load(self.root / "val_X.pt")
                val_y = torch.load(self.root / "val_y.pt")
                test_X = torch.load(self.root / "test_X.pt")
                test_y = torch.load(self.root / "test_y.pt")
            else:
                # We'll need torchaudio for processing, but skip implementation here
                # Using pre-processed data is recommended
                logger.warning("Raw data processing not implemented, please use pre-processed data")
                train_X = torch.randn(1, 16000, 1)  # Dummy data
                train_y = torch.zeros(1, dtype=torch.long)
                val_X = torch.randn(1, 16000, 1)
                val_y = torch.zeros(1, dtype=torch.long)
                test_X = torch.randn(1, 16000, 1)
                test_y = torch.zeros(1, dtype=torch.long)

            if not os.path.exists(data_loc):
                os.makedirs(data_loc, exist_ok=True)
            save_data(
                data_loc,
                train_X=train_X,
                val_X=val_X,
                test_X=test_X,
                train_y=train_y,
                val_y=val_y,
                test_y=test_y,
            )

        X, y = self.load_data(data_loc, partition) # (batch, length, 1)
        if self.gen: 
            y = y.transpose(1, 2)

        if not mfcc and not self.gen:
            X = F.pad(X, (0, 0, 0, length-16000))

        # Subsample
        if not mfcc:
            X, y = subsample(X, y, sr)

        if self.discrete_input:
            X = X.long().squeeze()

        self.src = X
        self.tgt = y

    def __getitem__(self, index):

        example = {
            "id": index,
            "source": self.src[index],
            "target": self.tgt[index],
        }

        return example

    def __len__(self):
        return len(self.src)

    def num_tokens(self, index):
        return len(self.src[index])

    def size(self, index):
        return len(self.src[index])

    def collater(self, samples):
        samples = [
            s
            for s in samples
            if s["source"] is not None
        ]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        targets = [s["target"] for s in samples]
        sizes = [len(s) for s in sources]

        def _collate(batch, resolution=1):
            # From https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
            elem = batch[0]
            if isinstance(elem, torch.Tensor):
                out = None
                x = torch.stack(batch, dim=0, out=out)
                if resolution is not None:
                    x = x[:, ::resolution] # assume length is first axis after batch
                return x
            else:
                batch = torch.tensor(batch)
                if resolution is not None:
                    batch = batch[:, ::resolution] # assume length is first axis after batch
                return batch

        src_lengths = torch.LongTensor(sizes)
        src_tokens = _collate(sources, resolution=self.resolution)
        src_tokens = src_tokens.squeeze(-1)
        target = _collate(targets, resolution=None)

        ntokens = src_lengths.sum().item()

        id = torch.LongTensor([s["id"] for s in samples])

        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
            'target': target,
        }

        return batch

    @staticmethod
    def load_data(data_loc, partition):
        tensors = load_data(data_loc)
        if partition == "train":
            X = tensors["train_X"]
            y = tensors["train_y"]
        elif partition == "val":
            X = tensors["val_X"]
            y = tensors["val_y"]
        elif partition == "test":
            X = tensors["test_X"]
            y = tensors["test_y"]
        else:
            raise NotImplementedError("the set {} is not implemented.".format(partition))

        return X.transpose(1, 2), y


class SCTruncateDataset(BaseWrapperDataset):
    """Truncate a sequence by returning the first truncation_length tokens
    tailored for speech command dataset
    """

    def __init__(self, dataset, truncation_length):
        super().__init__(dataset)
        assert truncation_length is not None
        self.truncation_length = truncation_length
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        item_len = len(item['source'])
        if item_len > self.truncation_length:
            item['source'] = item['source'][:self.truncation_length]
        return item

    @property
    def sizes(self):
        return np.minimum(self.dataset.sizes, self.truncation_length)

    def __len__(self):
        return len(self.dataset)