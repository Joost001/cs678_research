import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score

import time, random, numpy as np, argparse, sys, re, os

from CovostDataset import CovostDataset
from encoder_decoder_new import CustomTransformer, create_mask #, make_tgt_mask

import ipdb

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#based on https://pytorch.org/tutorials/beginner/translation_transformer.html
def greedy_decode(model, src, tgt, src_mask, max_len, bos_id=1, eos_id=2):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    
    # ipdb.set_trace()
    memory = model.encoder(model.positional_encoding_enc(src), src_mask)
    memory = memory.to(DEVICE)
    # memory = model.encoder(model.positional_encoding(src), src_mask)
    
    output_tokens = torch.ones(src.size(0),1).fill_(bos_id).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        _, tgt_padding_mask, tgt_mask = create_mask(src, output_tokens, 0)
        tgt_padding_mask, tgt_mask = tgt_padding_mask.to(DEVICE), tgt_mask.to(DEVICE)
        
        #Use the previous names
        out = model.positional_encoding_dec(model.emb_dec(output_tokens))
        out = model.decoder(out, memory, tgt_padding_mask, tgt_mask)
        
        # out = model.positional_encoding(model.tok_emb(output_tokens))
        # out = model.decoder(out, memory, tgt_padding_mask, tgt_mask)
        
        prob = model.generator(out[:,-1, :])
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.type(torch.LongTensor)
        
        output_tokens = torch.cat([output_tokens, next_word.view(-1,1)], dim=1)
    
    return output_tokens

def model_eval(dataloader, tokenizer, model, padding_indx, loss_fn):
    model.eval() # switch to eval model, will turn off randomness like dropout
    
    loss_batches = 0
    num_batches = 0
    candidate_corpus = []
    reference_corpus = []
    
    for i_batch, sample_batched in enumerate(dataloader):
        b_waves, b_frames, b_texts, b_speakers = sample_batched[0]['fbank_waves'], sample_batched[0]['n_frames'], \
                                                        sample_batched[0]['tgt_texts'], sample_batched[0]['speakers']

        b_waves = b_waves.to(DEVICE)
        tgt = torch.tensor(b_texts).to(DEVICE)
        tgt_input = tgt[:, :-1]
        
        src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(b_waves, tgt_input, padding_indx)
        src_padding_mask, tgt_padding_mask, tgt_mask = src_padding_mask.to(DEVICE), tgt_padding_mask.to(DEVICE), tgt_mask.to(DEVICE)
        
        
        logits = model(b_waves, tgt_input, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask, tgt_mask=tgt_mask)
        tgt_out = tgt[:, 1:]
        
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss_batches += loss.item()
        num_batches += 1
            
        texts = greedy_decode(model, b_waves, tgt_input, src_padding_mask, tgt_input.size(dim=1)+5, bos_id=1, eos_id=2)
        
        # ipdb.set_trace()
        for i in range(len(b_texts)):
            candidate_corpus.append(tokenizer.sp.decode_ids([word.item() for word in texts[i]]).replace('.',' ').split())
            reference_corpus.append(tokenizer.sp.decode_ids(b_texts[i]).replace('.',' ').split())
    
    # ipdb.set_trace()    
    bleu= bleu_score(candidate_corpus, reference_corpus, max_n=4)
    ave = loss_batches/num_batches
    
    return bleu, ave

def train(args):
    train_tsv_root = f'/datasets/CS678/{args.src_lan}/train_st_{args.src_lan}_{args.tgt_lan}.tsv'
    
    args.vocab_size = args.vocab_size+4
    
    zip_fbank_path = f'/datasets/CS678/{args.src_lan}/fbank80.zip'
    corpus_path = f'/datasets/CS678/{args.src_lan}/corpus_train_{args.src_lan}_{args.tgt_lan}.txt' #f'/scratch/jvasqu6/CS678/Project/corpus/{args.src_lan}/corpus_train_{args.src_lan}_{args.tgt_lan}.txt'

    train_dataset = CovostDataset(tsv_root=train_tsv_root, zip_fbank_path=zip_fbank_path, corpus_path=corpus_path, args=args)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)
    
    model = CustomTransformer(embed_size=args.embed_size,
                                heads=args.num_heads,
                                inner_layer_size=args.inner_layer_size,
                                n_blocks=args.num_blocks,
                                vocab_size=args.vocab_size) #4 for the 4 tokens: BOS, EOS, UNK, PAD

    model = model.to(DEVICE)
    
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
    
            b_waves = b_waves.to(DEVICE)
            tgt = torch.tensor(b_texts).to(DEVICE)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            
            # ipdb.set_trace()
            src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(b_waves, tgt_input, padding_indx)
            src_padding_mask, tgt_padding_mask, tgt_mask = src_padding_mask.to(DEVICE), tgt_padding_mask.to(DEVICE), tgt_mask.to(DEVICE)
            
            # ipdb.set_trace()
            torch.autograd.set_detect_anomaly(True)
            optimizer.zero_grad()
            logits = model(b_waves, tgt_input, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask, tgt_mask=tgt_mask)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
    
            optimizer.step()
            train_batch_loss += loss.item()
            num_batch_batches += 1 
            
            train_batches += loss.item()
            num_batches += 1
            
            texts = greedy_decode(model, b_waves, tgt_input, src_padding_mask, tgt_input.size(dim=1)+5, bos_id=1, eos_id=2)
            
            # ipdb.set_trace()
            for i in range(len(b_texts)):
                candidate_corpus.append(train_dataset.tokenizer.sp.decode_ids([word.item() for word in texts[i]]).lower().replace('.',' ').split())
                reference_corpus.append(train_dataset.tokenizer.sp.decode_ids(b_texts[i]).lower().replace('.',' ').split())
            
            if (i_batch + 1) % print_each == 0:
                ave = train_batch_loss/num_batch_batches
                print(f"Epoch {epoch}, batch {i_batch}, batch loss ave:: {ave : .3f}")
                train_batch_loss = 0
                num_batch_batches = 0
        
        # ipdb.set_trace()  
        bleu_train= bleu_score(candidate_corpus, reference_corpus, max_n=5)
        ave = train_batches/num_batches
        
        print(f"Overall Epoch {epoch}: train loss :: { ave:.3f}, train bleu :: { bleu_train:.3f}")
        print()
        
        train_losts += [ave]
        
        train_bleus += [bleu_train]
    
        path = f'/scratch/jvasqu6/CS678/Project/checkpoints/{args.src_lan}/model_{args.src_lan}_{args.tgt_lan}_ep_{epoch}.pt'
        
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_train': train_losts,
                'bleu_train': train_bleus,
                }, path)
                
def evaluate(args):
    test_tsv_root = f'/datasets/CS678/{args.src_lan}/test_st_{args.src_lan}_{args.tgt_lan}.tsv'
    dev_tsv_root = f'/datasets/CS678/{args.src_lan}/dev_st_{args.src_lan}_{args.tgt_lan}.tsv'
    
    args.vocab_size = args.vocab_size+4
    
    zip_fbank_path = f'/datasets/CS678/{args.src_lan}/fbank80.zip'
    corpus_path = f'/scratch/jvasqu6/CS678/Project/corpus/{args.src_lan}/corpus_train_{args.src_lan}_{args.tgt_lan}.txt' #f'/datasets/CS678/{args.src_lan}/corpus_train_{args.src_lan}_{args.tgt_lan}.txt'

    dev_dataset = CovostDataset(tsv_root=dev_tsv_root, zip_fbank_path=zip_fbank_path, corpus_path=corpus_path, args=args)
    dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)
    
    test_dataset = CovostDataset(tsv_root=test_tsv_root, zip_fbank_path=zip_fbank_path, corpus_path=corpus_path, args=args)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)
    
    padding_indx = 0
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=padding_indx)
    
    dev_losts = []
    test_losts = []
    
    dev_bleus = []
    test_bleus = []
    
    print('Evaluating Translator')
    for epoch in range(args.epochs):
        model = CustomTransformer(embed_size=args.embed_size,
                                heads=args.num_heads,
                                inner_layer_size=args.inner_layer_size,
                                n_blocks=args.num_blocks,
                                vocab_size=args.vocab_size) #4 for the 4 tokens: BOS, EOS, UNK, PAD

        model = model.to(DEVICE)
        
        if torch.cuda.is_available():
            checkpoint = torch.load(f'/scratch/jvasqu6/CS678/Project/checkpoints/{args.src_lan}/model_{args.src_lan}_{args.tgt_lan}_ep_{epoch}.pt')
        else:
            checkpoint = torch.load(f'/scratch/jvasqu6/CS678/Project/checkpoints/{args.src_lan}/model_{args.src_lan}_{args.tgt_lan}_ep_{epoch}.pt',  map_location=torch.device('cpu'))
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        bleu_dev, loss_dev = model_eval(dev_dataloader, dev_dataset.tokenizer, model, padding_indx, loss_fn)
        bleu_test, loss_test = model_eval(test_dataloader, test_dataset.tokenizer, model, padding_indx, loss_fn)
        
        print(f"Overall Epoch {epoch}: dev loss :: {loss_dev:.3f}, test loss :: {loss_test:.3f}")
        print(f"Overall Epoch {epoch}: dev bleu :: {bleu_dev:.3f}, test bleu :: {bleu_test:.3f}")
        print()
        
        dev_losts += [loss_dev]
        test_losts += [loss_test]
        
        dev_bleus = [bleu_dev]
        test_bleus = [bleu_test]
    
        path = f'/scratch/jvasqu6/CS678/Project/checkpoints/{args.src_lan}/model_eval_{args.src_lan}_{args.tgt_lan}_ep_{epoch}.pt'
        
        torch.save({
                'epoch': epoch,
                'loss_dev': dev_losts,
                'loss_test': test_losts,
                'bleu_dev': dev_bleus,
                'bleu_test': test_bleus,
                }, path)
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--src_lan", type=str, default='es', help='source language (can be en, es, fr, fa, ca)')
    parser.add_argument("--tgt_lan", type=str, default='en', help='target language (can be en, ca, fa)')
    parser.add_argument("--vocab_size", type=int, default=2000, help='size of vocab')
    parser.add_argument("--epochs", type=int, default=20, help='number of epochs')
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--train_only', action='store_true')

    parser.add_argument('--embed_size', type=int, default=80, help='embedding size')
    parser.add_argument("--num_heads", type=int, default = 8, help='number of heads for multi attention heads, default 8')
    parser.add_argument("--num_blocks", type=int, default = 5, help='number of blocks in the decoder and encoder, default 5')
    parser.add_argument("--inner_layer_size", type=int, default = 256, help='inner layer size, default 512')
    parser.add_argument("--dropout", type=float, default = .2, help='dropout, default 0.2')
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    return args

if __name__ == "__main__":
    args = get_args()
    
    if not args.eval_only:
        print('Training...')
        train(args)
    if not args.train_only:
        print('Evaluating...')
        evaluate(args)
    
