import torch
import torchaudio

#General
import pandas as pd
from pathlib import Path
from utils import (
    load_df_from_tsv,
    get_fbank_wave_from_zip
    )
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class CovostDataset(Dataset):
    def __init__(self, tsv_root: str, zip_fbank_path: str, args):
        self.dataset = load_df_from_tsv(tsv_root)
        self.zip_fbank_path = zip_fbank_path
        self.args = args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        '''
        Return the fbank_wave, n_frames, tgt_text, speaker according to idx
        idx must coincides with id of self.dataset['id']
        '''
        sample = self.dataset[dataset['id']==idx]
        n_frames = sample['n_frames']
        tgt_text = sample['tgt_text']
        speaker = sample['speaker']
        fbank_wave = get_fbank_wave_from_zip(self.zip_fbank_path,sample['audio'])
        
        sample = {'fbank_wave': fbank_wave,
                  'n_frames': n_frames,
                  'tgt_text': tgt_text,
                  'speaker': speaker}
        
        return sample
