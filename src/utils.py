"""
Based on the fairseq repo: https://github.com/facebookresearch/fairseq
"""
import csv
import io
import zipfile
from functools import reduce
from pathlib import Path
from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torchaudio.sox_effects as ta_sox
from tqdm import tqdm
import torchaudio.compliance.kaldi as ta_kaldi


def load_df_from_tsv(path):
    """
    # todo insert desc of function
    :Note: Union[str, Path] is equal to str | Path
    :param path: (Union[str, Path]):
    :return: (pd.DataFrame):
    """
    # as_posix() Return a string representation of the path with forward slashes (/). Ex: p.as_posix()->'c:/windows'
    _path = path if isinstance(path, str) else path.as_posix()
    return pd.read_csv(_path, sep='\t', header=0, encoding='utf-8', escapechar='\\', quoting=csv.QUOTE_NONE,
                       na_filter=False)


def extract_fbank_features(waveform, sample_rate, output_path=None, n_mel_bins=80, overwrite=False):
    """
    # todo insert desc of function
    :param waveform: (torch.FloatTensor):
    :param sample_rate: (int):
    :param output_path: (Optional[Path]):
    :param n_mel_bins: (int):
    :param overwrite: (bool):
    :return:
    """

    if output_path is not None and output_path.is_file() and not overwrite:
        return

    _waveform, _ = convert_waveform(waveform, sample_rate, to_mono=True)
    # Kaldi compliance: 16-bit signed integers
    _waveform = _waveform * (2 ** 15)
    _waveform = _waveform.numpy()

    features = _get_torchaudio_fbank(_waveform, sample_rate, n_mel_bins)
    if features is None:
        raise ImportError("Please install pyKaldi or torchaudio to enable fbank feature extraction")

    if output_path is not None:
        np.save(output_path.as_posix(), features)

    return features


def convert_waveform(waveform, sample_rate, normalize_volume=False, to_mono=False, to_sample_rate=None):
    """
    convert a waveform to a target sample rate | from multichannel to mono channel | volume normalization
    :param waveform: (Union[np.ndarray, torch.Tensor]): 2D original waveform (channels x length)
    :param sample_rate: (int): original sample rate
    :param normalize_volume: (bool): perform volume normalization
    :param to_mono: (bool): convert to mono channel if having multiple channels
    :param to_sample_rate: (int): target sample rate
    :return: (Tuple[Union[np.ndarray, torch.Tensor], int]): waveform converted 2D waveform (channels x length), sample_rate (float): target sample rate
    """
    effects = []
    if normalize_volume:
        effects.append(["gain", "-n"])
    if to_sample_rate is not None and to_sample_rate != sample_rate:
        effects.append(["rate", f"{to_sample_rate}"])
    if to_mono and waveform.shape[0] > 1:
        effects.append(["channels", "1"])
    if len(effects) > 0:
        is_np_input = isinstance(waveform, np.ndarray)
        # _waveform = torch.from_numpy(waveform) if is_np_input else waveform
        if is_np_input:
            _waveform = torch.from_numpy(waveform)
        else:
            _waveform = waveform
        converted, converted_sample_rate = ta_sox.apply_effects_tensor(_waveform, sample_rate, effects)
        if is_np_input:
            converted = converted.numpy()
        return converted, converted_sample_rate
    return waveform, sample_rate


def _get_torchaudio_fbank(waveform, sample_rate, n_bins=80):
    """
    Get mel-filter bank features via TorchAudio
    :param waveform: (np.ndarray):
    :param sample_rate:
    :param n_bins:
    :return: (np.ndarray)
    """
    waveform = torch.from_numpy(waveform)
    features = ta_kaldi.fbank(waveform, num_mel_bins=n_bins, sample_frequency=sample_rate)
    return features.numpy()


def filter_manifest_df(df, is_train_split=False, extra_filters=None, min_n_frames=5, max_n_frames=3000):
    """
    # todo insert desc of function
    :param df:
    :param is_train_split:
    :param extra_filters:
    :param min_n_frames:
    :param max_n_frames:
    :return:
    """
    filters = {"no speech": df["audio"] == "", f"short speech (<{min_n_frames} frames)": df["n_frames"] < min_n_frames,
                "empty sentence": df["tgt_text"] == ""}

    if is_train_split:
        filters[f"long speech (>{max_n_frames} frames)"] = df["n_frames"] > max_n_frames
    if extra_filters is not None:
        filters.update(extra_filters)

    invalid = reduce(lambda x, y: x | y, filters.values())
    valid = ~invalid
    print("| " + ", ".join(f"{n}: {f.sum()}" for n, f in filters.items())
          + f", total {invalid.sum()} filtered, {valid.sum()} remained.")
    return df[valid]


def save_df_to_tsv(dataframe, path):
    """
    # todo insert desc of function
    :param dataframe:
    :param path: (Union[str, Path]):
    :return: None
    """
    _path = path if isinstance(path, str) else path.as_posix()
    dataframe.to_csv(_path, sep="\t", header=True, index=False, encoding="utf-8", escapechar="\\",
                     quoting=csv.QUOTE_NONE)


def create_zip(data_root, zip_path):
    """
    # todo insert desc of function
    :param data_root: (Path):
    :param zip_path: (Path):
    :return:
    """
    paths = list(data_root.glob("*.npy"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as f:
        for path in tqdm(paths):
            f.write(path, arcname=path.name)


def get_zip_manifest(zip_path, zip_root = None):
    """
    # todo insert desc of function
    :param zip_path: (Path):
    :param zip_root: (Optional[Path]):
    :return:
    """
    _zip_path = Path.joinpath(zip_root or Path(""), zip_path)
    with zipfile.ZipFile(_zip_path, mode="r") as f:
        info = f.infolist()
    paths, lengths = {}, {}
    for i in tqdm(info):
        utt_id = Path(i.filename).stem
        offset, file_size = i.header_offset + 30 + len(i.filename), i.file_size
        paths[utt_id] = f"{zip_path.as_posix()}:{offset}:{file_size}"
        with open(_zip_path, "rb") as f:
            f.seek(offset)
            byte_data = f.read(file_size)
            assert len(byte_data) > 1
            assert is_npy_data(byte_data), i
            byte_data_fp = io.BytesIO(byte_data)
            lengths[utt_id] = np.load(byte_data_fp).shape[0]
    return paths, lengths


def get_fbank_wave_from_zip(zip_path, manifest_path, zip_root=None):
    """
    Return the fbank from the zip in numpy format
    :param zip_path: (Path):
    :param manifest_path: (Path):
    :param zip_root: Optional[Path]:
    :return:
    """
    _zip_path = Path.joinpath(zip_root or Path(""), zip_path)
    _path,  offset, file_size = str(manifest_path).split(':')
    offset, file_size = int(offset), int(file_size)
    
    with open(_zip_path, "rb") as f:
        f.seek(offset)
        byte_data = f.read(file_size)
        byte_data_fp = io.BytesIO(byte_data)
        fbank_wave = np.load(byte_data_fp)
    return fbank_wave


def is_npy_data(data):
    """
    # todo insert desc of function
    :param data: (bytes):
    :return: (bool):
    """
    return data[0] == 147 and data[1] == 78
