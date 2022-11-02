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
corpus_path = '/datasets/CS678/es/corpus_train_es_en.txt'

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--src_lan", type=str, default='es', help='source language (can be en, es, fr, fa, ca)')
parser.add_argument("--tgt_lan", type=str, default='en', help='target language (can be en, ca, fa)')
parser.add_argument("--vocab_size", type=int, default=2000, help='size of vocab')
args = parser.parse_args()

train_dataset = CovostDataset(tsv_root=tsv_root, zip_fbank_path=zip_fbank_path, corpus_path=corpus_path, args=args)


train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)

for i_batch, sample_batched in enumerate(train_dataloader):
    b_waves, b_frames, b_texts, b_speakers = sample_batched[0]['fbank_waves'], sample_batched[0]['n_frames'], \
                                                sample_batched[0]['tgt_texts'], sample_batched[0]['speakers']
    
    print('Example of decoding from ids to text:')
    print(f'Ids: {b_texts[0]}')
    print(f'Texts: {train_dataset.tokenizer.sp.decode_ids(b_texts[0])}')
    
    print(f'batch number: {i_batch} done')
    
    if i_batch==4:
        break