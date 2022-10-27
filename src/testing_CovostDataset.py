import pandas as pd
import numpy as np
from pathlib import Path
from CovostData import CovostDataset
import torch

from utils import load_df_from_tsv

from torch.utils.data import Dataset, DataLoader

tsv_root = '/datasets/CS678/es/train_st_es_en.tsv'
zip_fbank_path = '/datasets/CS678/es/fbank80.zip'
train_dataset = CovostDataset(tsv_root=tsv_root, zip_fbank_path=zip_fbank_path)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=60)

for i_batch, sample_batched in enumerate(train_dataloader):
    b_waves, b_frames, b_texts, b_speakers = sample_batched['fbank_wave'], sample_batched['n_frames'], sample_batched['tgt_text'] /
                                            sample_batched['speaker']
    
    print(f'batch number: {i}')
    print('waves')
    print(b_waves)
    print()
    print('frames')
    print(b_frames)
    print()
    print('texts')
    print(b_texts)
    print()
    print('speakers')
    print(b_speakers)
    
    if i_batch==4:
        break