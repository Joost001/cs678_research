import torch
from torch.utils.data import Dataset, DataLoader

import time, random, numpy as np, argparse, sys, re, os
import pandas as pd
import numpy as np
from pathlib import Path

from utils import load_df_from_tsv
from CovostDataset import CovostDataset

tsv_root = '/datasets/CS678/es/train_st_es_en.tsv'
zip_fbank_path = '/datasets/CS678/es/fbank80.zip'

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
args = parser.parse_args()

train_dataset = CovostDataset(tsv_root=tsv_root, zip_fbank_path=zip_fbank_path, args=args)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)

for i_batch, sample_batched in enumerate(train_dataloader):
    b_waves, b_frames, b_texts, b_speakers = sample_batched[0]['fbank_waves'], sample_batched[0]['n_frames'], \
                                                sample_batched[0]['tgt_texts'], sample_batched[0]['speakers']
    
    print(f'batch number: {i_batch} done')
    
    if i_batch==4:
        break