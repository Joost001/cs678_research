## Based on the fairseq repo: https://github.com/facebookresearch/fairseq
#Torch
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torchaudio.datasets.utils import download_url, extract_archive
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

#General
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path
from utils import load_df_from_tsv, extract_fbank_features, filter_manifest_df, save_df_to_tsv, create_zip, get_zip_manifest
import argparse
from tqdm import tqdm
import os
import io
os.environ['KMP_DUPLICATE_LIB_OK']='True'

COLUMNS = ['id', 'audio', 'n_frames', 'tgt_text', 'speaker']

class CoVoST(Dataset):
    '''
    Create a dataset for CoVoST version 4.0

    Inputs:
    root (str): root path to the dataset and generated features
    split (str): split from datasets (train, test or dev)
    source_language (str): source (audio) langauge
    target_language (str): target (text) language
    '''
    SPLITS = ['train', 'dev', 'test']
    
    #Available pair of translations
    XX_EN_LANGUAGES=['fr', 'es', 'ca', 'fa']
    EN_XX_LANGUAGES = ['ca', 'fa']

    #Translations
    COVOST_URL_TEMPLATE = (
        "https://dl.fbaipublicfiles.com/covost/"
        "covost_v2.{src_lang}_{tgt_lang}.tsv.tar.gz"
    )

    def __init__(self, root: str, split: str, source_language: str, target_language: str) -> None:
        # Assert if given source and target languages are correct
        assert 'en' in {source_language, target_language}
        if source_language == 'en':
            assert target_language in self.EN_XX_LANGUAGES
        else:
            assert source_language in self.XX_EN_LANGUAGES
        
        # Retrieve path where are the files, then assert if it contains the files
        self.root = Path(root)
        cv_tsv_path = self.root / 'validated.tsv'
        assert cv_tsv_path.is_file()

        # Download translations in case they are not in the path
        covost_url = self.COVOST_URL_TEMPLATE.format(src_lang=source_language, tgt_lang = target_language)
        covost_archive = self.root / Path(covost_url).name.replace('.tar.gz', '')
        # Try first with the .tsv file then with the .tar.gz file
        if not covost_archive.is_file():
            covost_archive = self.root / Path(covost_url).name
            if not covost_archive.is_file():
                download_url(covost_url, self.root.as_posix(), hash_value=None)
            extract_archive(covost_archive.as_posix())

        # Merge the tsv files from the source lang with translation from the target lang
        cv_tsv = load_df_from_tsv(cv_tsv_path)
        covost_tsv = load_df_from_tsv(self.root / Path(covost_url).name.replace('.tar.gz', ''))

        df = pd.merge(
            left = cv_tsv[['path', 'sentence', 'client_id']],
            right = covost_tsv[['path', 'translation', 'split']],
            how = 'inner',
            on = 'path'
        )

        # Obtain the split required (train, dev, test)
        # For some reason, in validated and downloaded translations there are some examples with the
        # label 'train_covost'. In the paper these examples are also included in the training 
        assert split in self.SPLITS
        if split=='train':
            df = df[(df['split'] == split) | (df['split'] == f'{split}_covost')]
        else:
            df = df[df['split'] == split]

        data = df.to_dict(orient = 'index').items()
        data = [v for k, v in sorted(data, key=lambda x:x[0])]
        self.data =[]
        for e in data:
            try:
                path = self.root / 'clips' / e['path']
                _ = torchaudio.info(path.as_posix())
                self.data.append(e)
            except RuntimeError:
                pass
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, Optional[str], str, str]:
        '''
        Load and return the n-th sample from the dataset.

        Inputs:
        n (int): index of sample to retrieve

        Outputs:
        tuple: (waveform, sample_rate, sentence, translation, speaker_id, sample_id)
        '''
        data = self.data[n]
        path = self.root / 'clips' / data['path']
        waveform, sample_rate = torchaudio.load(path)
        waveform = torch.tensor(waveform)
        sentence = data['sentence']
        translation = data['translation']
        speaker_id = data['client_id']
        _id = data['path'].replace('.mp3', '')
        return waveform, sample_rate, sentence, translation, speaker_id, _id


# Note: to retrieve a batch instead of one item, you could use an iterator with DataLoader
# import itertools
# loader = iter(DataLoader(CoVoST(root, split, source_language, target_language),
#                           batch_size=bs, 
#                           shuffle=True))


def process(args):
    '''
    Process the dataset
    '''
    print('processing')
    # Retreive the root folder that contains the audio from source langauge
    root = Path(args.data_root).absolute() / args.src_lang
    if not root.is_dir():
        raise NotADirectoryError(f"{root} does not exist")
    
    # Extract features by using the required preprocessing procedures
    print('starting features extraction')
    feature_root = root / "fbank80"
    feature_root.mkdir(exist_ok=True)
    for split in CoVoST.SPLITS:
        print(f"Fetching split {split}...")
        dataset = CoVoST(root, split, args.src_lang, args.tgt_lang)
        print("Extracting log mel filter bank features...")
        for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
            extract_fbank_features(
                waveform, sample_rate, feature_root / f"{utt_id}.npy"
            )

    # Pack features into ZIP
    zip_path = root / "fbank80.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_path)

    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    task = f"st_{args.src_lang}_{args.tgt_lang}"
    for split in CoVoST.SPLITS:
        manifest = {c: [] for c in COLUMNS}
        dataset = CoVoST(root, split, args.src_lang, args.tgt_lang)
        for _, _, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
            manifest["id"].append(utt_id)
            manifest["audio"].append(audio_paths[utt_id])
            manifest["n_frames"].append(audio_lengths[utt_id])
            manifest["tgt_text"].append(src_utt if args.tgt_lang is None else tgt_utt)
            manifest["speaker"].append(speaker_id)
        is_train_split = split.startswith("train")
        if is_train_split:
            train_text.extend(manifest["tgt_text"])
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(df, is_train_split=is_train_split)
        save_df_to_tsv(df, root / f"{split}_{task}.tsv")

if __name__ == '__main__':
    print('Reading arguments...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str, help="data root with sub-folders for each language <root>/<src_lang>")
    parser.add_argument("--src-lang", "-s", required=True, type=str)
    parser.add_argument("--tgt-lang", "-t", type=str)
    
    args = parser.parse_args()

    print('Starting preprocessing...')
    process(args)
