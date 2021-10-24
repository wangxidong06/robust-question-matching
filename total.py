from functools import partial
import argparse
import os
import random
import time

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup

from work.data import create_dataloader, read_text_pair, convert_example

