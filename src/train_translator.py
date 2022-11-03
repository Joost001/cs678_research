import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score


import time, random, numpy as np, argparse, sys, re, os
import pandas as pd
from pathlib import Path

from utils import load_df_from_tsv
from CovostDataset import CovostDataset
from encoder_decoder import CustomTransformer, create_mask

import ipdb

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def model_eval(dataloader, tokenizer, model, padding_indx, loss_fn, device):
    model.eval() # switch to eval model, will turn off randomness like dropout
    # device = torch.device("cpu")
    # model.to(device)
    
    train_batches = 0
    num_batches = 0
    candidate_corpus = []
    reference_corpus = []
    
    
    for i_batch, sample_batched in enumerate(dataloader):
        b_waves, b_frames, b_texts, b_speakers = sample_batched[0]['fbank_waves'], sample_batched[0]['n_frames'], \
                                                        sample_batched[0]['tgt_texts'], sample_batched[0]['speakers']

        b_waves = b_waves.to(device)
        tgt = torch.tensor(b_texts).to(device)
        tgt_input = tgt[:, :-1]
        
        src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(b_waves, tgt_input, padding_indx)
        src_padding_mask, tgt_padding_mask, tgt_mask = src_padding_mask.to(device), tgt_padding_mask.to(device), tgt_mask.to(device)

        logits = model(b_waves, tgt_input, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask, tgt_mask=tgt_mask)
        tgt_out = tgt[:, 1:]
        
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        train_batches += loss.item()
        num_batches += 1
            
        pbb = F.softmax(logits, dim=-1)
        texts = torch.argmax(pbb, dim=-1).type(torch.LongTensor)
        
        
        for i in range(len(b_texts)):
            candidate_corpus.append(tokenizer.sp.decode_ids([word.item() for word in texts[i]]).replace('.',' ').split())
            reference_corpus.append(tokenizer.sp.decode_ids(b_texts[i]).replace('.',' ').split())
    
    # ipdb.set_trace()    
    bleu= bleu_score(candidate_corpus, reference_corpus, max_n=4)
    ave = train_batches/num_batches
    
    return bleu, ave

def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_tsv_root = f'/datasets/CS678/{args.src_lan}/train_st_{args.src_lan}_{args.tgt_lan}.tsv'
    test_tsv_root = f'/datasets/CS678/{args.src_lan}/test_st_{args.src_lan}_{args.tgt_lan}.tsv'
    dev_tsv_root = f'/datasets/CS678/{args.src_lan}/dev_st_{args.src_lan}_{args.tgt_lan}.tsv'
    
    args.vocab_size = args.vocab_size+4
    
    zip_fbank_path = f'/datasets/CS678/{args.src_lan}/fbank80.zip'
    corpus_path = f'/scratch/jvasqu6/CS678/Project/corpus/{args.src_lan}/corpus_train_{args.src_lan}_{args.tgt_lan}.txt' #f'/datasets/CS678/{args.src_lan}/corpus_train_{args.src_lan}_{args.tgt_lan}.txt'

    train_dataset = CovostDataset(tsv_root=train_tsv_root, zip_fbank_path=zip_fbank_path, corpus_path=corpus_path, args=args)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)
    
    test_dataset = CovostDataset(tsv_root=test_tsv_root, zip_fbank_path=zip_fbank_path, corpus_path=corpus_path, args=args)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)
    
    dev_dataset = CovostDataset(tsv_root=dev_tsv_root, zip_fbank_path=zip_fbank_path, corpus_path=corpus_path, args=args)
    dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

    model = CustomTransformer(embed_size=args.embed_size,
                                heads=args.num_heads,
                                inner_layer_size=args.inner_layer_size,
                                n_blocks=args.num_blocs,
                                vocab_size=args.vocab_size) #4 for the 4 tokens: BOS, EOS, UNK, PAD

    model = model.to(device)
    
    padding_indx = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=padding_indx)
    
    train_losts = []
    dev_losts = []
    test_losts = []
    
    train_bleus = []
    dev_bleus = []
    test_bleus = []
    
    print('Training Translator')
    for epoch in range(args.epochs):
        model.train()
        # model.to(device)
        # print(next(model.parameters()).is_cuda)
    
        train_batches = 0
        num_batches = 0
        
        train_batch_loss = 0
        num_batch_batches = 0
            
        data_len = len(train_dataset.dataset)
        #Print results after training 10% of dataset
        print_each = int(data_len*0.1/args.batch_size)
        
        candidate_corpus = []
        reference_corpus = []
    
        for i_batch, sample_batched in enumerate(train_dataloader):
            b_waves, b_frames, b_texts, b_speakers = sample_batched[0]['fbank_waves'], sample_batched[0]['n_frames'], \
                                                        sample_batched[0]['tgt_texts'], sample_batched[0]['speakers']
    
            b_waves = b_waves.to(device)
            tgt = torch.tensor(b_texts).to(device)
            tgt_input = tgt[:, :-1]
    
            src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(b_waves, tgt_input, padding_indx)
            src_padding_mask, tgt_padding_mask, tgt_mask = src_padding_mask.to(device), tgt_padding_mask.to(device), tgt_mask.to(device)
    
            logits = model(b_waves, tgt_input, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask, tgt_mask=tgt_mask)
    
            tgt_out = tgt[:, 1:]
            
            
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
    
            optimizer.step()
            train_batch_loss += loss.item()
            num_batch_batches += 1 
            
            train_batches += loss.item()
            num_batches += 1
            
            pbb = F.softmax(logits, dim=-1)
            texts = torch.argmax(pbb, dim=-1).type(torch.LongTensor)
            for i in range(len(b_texts)):
                candidate_corpus.append(train_dataset.tokenizer.sp.decode_ids([word.item() for word in texts[i]]).lower().replace('.',' ').split())
                reference_corpus.append(train_dataset.tokenizer.sp.decode_ids(b_texts[i]).lower().replace('.',' ').split())
            
            if (i_batch + 1) % print_each == 0:
                ave = train_batch_loss/num_batch_batches
                print(f"Epoch {epoch}, batch {i_batch}, batch loss ave:: {ave : .3f}")
                train_batch_loss = 0
                num_batch_batches = 0
        
        # ipdb.set_trace()  
        bleu_train= bleu_score(candidate_corpus, reference_corpus, max_n=4)
        ave = train_batches/num_batches
        
        # bleu_dev, loss_dev = model_eval(dev_dataloader, train_dataset.tokenizer, model, padding_indx, loss_fn, device)
        # bleu_test, loss_test = model_eval(test_dataloader, train_dataset.tokenizer, model, padding_indx, loss_fn, device)
        
        print(f"Overall Epoch {epoch}: train loss :: { ave:.3f}, train bleu :: { bleu_train:.3f}")
        # print(f"Overall Epoch {epoch}: train loss :: { ave:.3f}, dev loss :: {loss_dev:.3f}, test loss :: {loss_test:.3f}")
        # print(f"Overall Epoch {epoch}: train bleu :: { bleu_train:.3f}, dev loss :: {bleu_dev:.3f}, test loss :: {bleu_test:.3f}")
        print()
        
        train_losts += [ave]
        # dev_losts += [loss_dev]
        # test_losts += [loss_test]
        
        train_bleus += [bleu_train]
        # dev_bleus = [bleu_dev]
        # test_bleus = [bleu_test]
    
        path = f'/scratch/jvasqu6/CS678/Project/checkpoints/{args.src_lan}/model_{args.src_lan}_{args.tgt_lan}_ep_{epoch}.pt'
        
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_train': train_losts,
                # 'loss_dev': dev_losts,
                # 'loss_test': test_losts,
                'bleu_train': train_bleus,
                # 'bleu_dev': dev_bleus,
                # 'bleu_test': test_bleus,
                }, path)
                
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--src_lan", type=str, default='es', help='source language (can be en, es, fr, fa, ca)')
    parser.add_argument("--tgt_lan", type=str, default='en', help='target language (can be en, ca, fa)')
    parser.add_argument("--vocab_size", type=int, default=2000, help='size of vocab')
    parser.add_argument("--epochs", type=int, default=20, help='number of epochs')
    parser.add_argument("--seed", type=int, default=11711)

    parser.add_argument('--embed_size', type=int, default=80, help='embedding size')
    parser.add_argument("--num_heads", type=int, default = 8, help='number of heads for multi attention heads, default 8')
    parser.add_argument("--num_blocs", type=int, default = 5, help='number of blocks in the decoder and encoder, default 4')
    parser.add_argument("--inner_layer_size", type=int, default = 256, help='inner layer size, default 512')
    parser.add_argument("--dropout", type=float, default = .2, help='dropout, default 0.2')
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    return args

if __name__ == "__main__":
    args = get_args()
    train(args)
    # test(args)