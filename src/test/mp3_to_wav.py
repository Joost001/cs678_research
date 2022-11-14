import argparse
import os
from preprocessing import ffmpeg, mkdir_if_dne
import multiprocessing as mp
from functools import partial

    
def convert_mp3_to_wav(in_f, input_dir, output_dir):
    failed_file=None
    try:
        base, ext = os.path.splitext(os.path.basename(in_f))
        base_out_f = base + ".wav"
        out_f = output_dir + "/" + base_out_f
        in_f = input_dir + "/" + in_f
        if not os.path.isfile(out_f):
            ffmpeg(in_f, out_f)
        assert(os.path.isfile(out_f)), "Fail to write file to dir!"
    except:
        failed_file = in_f
        pass
    return failed_file


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language","--lang", "-l", type=str, choices=["ca", "fa", "en"], required=True, help="language folder to operate on")
    parser.add_argument("--source-dir", "-s", type=str, required=True, help="source root dir")
    parser.add_argument("--target-dir", "-t", type=str, required=True, help="target root dir")
    args = parser.parse_args()

    # set lang
    language = args.language

    # set input and output dir
    source_dir=args.source_dir + "/{lang}/clips".format(lang=language)
    target_dir=args.target_dir + "/{lang}/clips_wav".format(lang=language)
    mkdir_if_dne(target_dir)

    # get list of mp3 files
    lof =  os.listdir(source_dir)
    
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(partial(convert_mp3_to_wav, input_dir=source_dir, output_dir=target_dir), [t for t in lof])
    pool.close()

    
