import glob
import argparse
import pandas as pd
from tqdm import tqdm


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language","--lang", "-l", type=str, choices=["ca", "fa", "en"], required=True, help="language folder to operate on")
    parser.add_argument("--target-dir", "-t", type=str, required=True, help="target root dir")
    args = parser.parse_args()
    
    # set lang
    language = args.language

    target_dir = args.target_dir + "/{lang}/*.tsv".format(lang=language)
    files = glob.glob(target_dir)
    for f in tqdm(files):
        df = pd.read_csv(f,sep="\t", quoting=3)
        if 'path' in df.columns:
            df['path'] = df['path'].str.replace('.mp3','.wav', regex=False)
            df.to_csv(f,sep='\t')
        else:
            print("file " + f + " does not have a path column")