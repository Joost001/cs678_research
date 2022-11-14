#!/usr/bin/env bash

# extract files
raw_dir=/datasets/CS678/joost/raw/
target_dir=/datasets/CS678/joost/
languages=("ca" "fa")

for l in "${languages[@]}"
do
    mkdir -p "$target_dir$l"
    tar -zvxf "${raw_dir}covost_v2.${l}_en.tsv.tar.gz" -C "$target_dir$l"
    tar -zvxf "$raw_dir$l.tar.tar" -C "$target_dir$l"
done

# covost splits
data_root=/datasets/CS678/joost/
py_get_covost=/home/jbottenb/repositories/covost/get_covost_splits.py

# XX to EN
python $py_get_covost --version 2 -s ca -t en -d $data_root/ca --cv-tsv $data_root/ca/validated.tsv
python $py_get_covost --version 2 -s fa -t en -d $data_root/fa --cv-tsv $data_root/fa/validated.tsv

# execute program mp3_to_wav.py
mp3_dir=/Users/joost/Data/
wav_dir=/Users/joost/Data/
python mp3_to_wav.py -l ca -s $mp3_dir -t $wav_dir
python mp3_to_wav.py -l fa -s $mp3_dir -t $wav_dir

# execute program ext_change.py
tsv_parent_dir=/datasets/CS678/joost
python ext_change.py -l ca -t $tsv_parent_dir
python ext_change.py -l fa -t $tsv_parent_dir

# create fbanks
# py_prep_covost=/home/jbottenb/repositories/fairseq/examples/speech_to_text/prep_covost_data.py
# python $py_prep_covost --data-root /Users/joost/Data --vocab-type bpe -s fa -t en --vocab-size 10000
# python $py_prep_covost --data-root /Users/joost/Data --vocab-type bpe -s ca -t en --vocab-size 10000
