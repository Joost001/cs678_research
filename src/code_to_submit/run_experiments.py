import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import time, random, numpy as np, argparse
import pandas as pd
import numpy as np
from CovostDataset import CovostDataset
from transfomer import PytorchTransformer


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, pad_idx, device):
    
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tsv_root = '/datasets/CS678/es/train_st_es_en.tsv'
zip_fbank_path = '/datasets/CS678/es/fbank80.zip'
corpus_path = '/datasets/CS678/es/corpus_train_es_en.txt'

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--src_lan", type=str, default='es', help='source language (can be en, es, fr, fa, ca)')
parser.add_argument("--tgt_lan", type=str, default='en', help='target language (can be en, ca, fa)')
parser.add_argument("--vocab_size", type=int, default=2000, help='size of vocab')
args = parser.parse_args()

train_dataset = CovostDataset(tsv_root=tsv_root, zip_fbank_path=zip_fbank_path, corpus_path=corpus_path, args=args)


train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)

# import ipdb
# ipdb.set_trace()
#  vocabs = [train_dataset.tokenizer.sp.id_to_piece(id) for id in range(train_dataset.tokenizer.sp.get_piece_size())]

embed_size = 80
heads = 20
inner_layer_size = 512
vocab_size = args.vocab_size
n_blocks = 5
padding_indx = 0

batch_size = 128
n_epochs = 50



model = PytorchTransformer(n_blocks, n_blocks, embed_size,
                                 heads, vocab_size, vocab_size, inner_layer_size)

model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, betas=(0.9, 0.98), eps=1e-9)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=padding_indx)

# import ipdb
# ipdb.set_trace()

model.train()

for epoch in range(n_epochs):
    
    losses = 0
    for i_batch, sample_batched in enumerate(train_dataloader):
        b_waves, b_frames, b_texts, b_speakers = sample_batched[0]['fbank_waves'], sample_batched[0]['n_frames'], \
            sample_batched[0]['tgt_texts'], sample_batched[0]['speakers']
            
        tgt = torch.tensor(b_texts)

        src = b_waves.permute(1, 0, 2).to(DEVICE)
        tgt = tgt.permute(1, 0).to(DEVICE)

        # import ipdb
        # ipdb.set_trace()

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src[:, :, 0], tgt_input, padding_indx, DEVICE) 

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)


        optimizer.zero_grad()
        
        tgt_out = tgt[1:, :]

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()  

        # if i_batch % 500 == 0:
        print(f"Epoch {epoch}, batch {i_batch}, loss {loss.item()}")


    print("---0---------------0--------------------0---------------0-------")
    print(f"Epoch {epoch}, loss {losses/len(train_dataloader)}\n")

    file_path = "/home/jmbuya/nlp/cs678_project/src/models"
    torch.save(model.state_dict(), f"{file_path}/model_lr_0.00001_e{epoch}_3.pt")

    # import ipdb
    # ipdb.set_trace()