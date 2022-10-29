import pandas as pd

from utils import load_df_from_tsv

XX_EN_LANGUAGES=['fr', 'es', 'ca', 'fa']
EN_XX_LANGUAGES = ['ca', 'fa']

for tgt in XX_EN_LANGUAGES:
    tsv = load_df_from_tsv(f'/datasets/CS678/{tgt}/train_st_{tgt}_en.tsv')    
    corpus = []
    corpus.extend(tsv['tgt_text'])

    f = open (f'/datasets/CS678/{tgt}/corpus_train_{tgt}_en.txt', 'w')
    for t in corpus:
        f.write(t + '\n')
    f.close()

for src in EN_XX_LANGUAGES:
    tsv = load_df_from_tsv(f'/datasets/CS678/en/train_st_en_{src}.tsv')    
    corpus = []
    corpus.extend(tsv['tgt_text'])

    f = open (f'/datasets/CS678/en/corpus_train_en_{src}.txt', 'w')
    for t in corpus:
        f.write(t + '\n')
    f.close()
