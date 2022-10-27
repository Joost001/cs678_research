import torch
import torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F

#General
import pandas as pd
import numpy as np
import argparse
import math
from pathlib import Path
from utils import (
    load_df_from_tsv,
    get_fbank_wave_from_zip
    )

class CovostDataset(Dataset):
    def __init__(self, tsv_root: str, zip_fbank_path: str, args):
        self.dataset = load_df_from_tsv(tsv_root)
        self.zip_fbank_path = zip_fbank_path
        self.p = args
        self.max_n_frame = 0
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        '''
        Return the fbank_wave, n_frames, tgt_text, speaker according to idx
        idx must coincides with id of self.dataset['id']
        '''
        sample = self.dataset.iloc[idx]
        n_frames = sample['n_frames']
        tgt_text = sample['tgt_text']
        speaker = sample['speaker']
        fbank_wave = get_fbank_wave_from_zip(self.zip_fbank_path,sample['audio'])
        
        sample = {'fbank_wave': fbank_wave,
                  'n_frames': n_frames,
                  'tgt_text': tgt_text,
                  'speaker': speaker}
        
        return sample
    
    def pad_data(self, data):
        frames = [x['n_frames'] for x in data]
        max_frames = max(frames)
        
        n_frames = np.array(frames)
        tgt_texts = np.array([x['tgt_text'] for x in data])
        speakers = np.array([x['speaker'] for x in data])
        
        fbank_waves = torch.empty((0, max_frames, data[0]['fbank_wave'].shape[1]))
        for i in range(len(data)):
            tensor_fbank_wave = torch.Tensor(data[i]['fbank_wave']).view(1,-1,data[0]['fbank_wave'].shape[1])
            total_padding = max_frames-data[i]['fbank_wave'].shape[0]
        
            # Hack: Padding size should be less than the corresponding input dimension
            while total_padding>=data[i]['fbank_wave'].shape[0]:
                tensor_fbank_wave = F.pad(tensor_fbank_wave, (0,0,0,data[i]['fbank_wave'].shape[0]-1), mode='reflect')
                total_padding -= (data[i]['fbank_wave'].shape[0]-1)
            
            tensor_fbank_wave = F.pad(tensor_fbank_wave, (0,0,0,total_padding), mode='reflect')
            fbank_waves = torch.cat((fbank_waves, tensor_fbank_wave), 0)
        
        return fbank_waves, n_frames, tgt_texts, speakers
        
    def collate_fn(self, all_data):
        all_data.sort(key = lambda x: -x['n_frames'])
        
        batches = []
        num_batches = int(np.ceil(len(all_data) / self.p.batch_size))
        for i in range(num_batches):
            start_idx = i * self.p.batch_size
            data = all_data[start_idx: start_idx + self.p.batch_size]

            fbank_waves, n_frames, tgt_texts, speakers = self.pad_data(data)
            batches.append({
                'fbank_waves': fbank_waves,
                'n_frames': n_frames,
                'tgt_texts': tgt_texts,
                'speakers': speakers,
            })
        
        return batches
        
