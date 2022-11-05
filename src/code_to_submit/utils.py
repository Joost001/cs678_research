## Based on the fairseq repo: https://github.com/facebookresearch/fairseq

# Torch, Audio
import torch
import torchaudio
import torchaudio.sox_effects as ta_sox
from torchaudio.functional import melscale_fbanks
# import soundfile as sf
import torchaudio.compliance.kaldi as ta_kaldi

# General
import io
import csv
import zipfile
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functools import reduce
from typing import Union, Optional, Tuple


def load_df_from_tsv(path: Union[str, Path]) -> pd.DataFrame: #note that Union[str, Path] is equal to str | Path
    _path = path if isinstance(path, str) else path.as_posix() #as_posix() Return a string representation of the path with forward slashes (/). Ex: p.as_posix()->'c:/windows'
    return pd.read_csv(_path, sep='\t', header=0, encoding='utf-8', escapechar='\\', quoting=csv.QUOTE_NONE, na_filter=False)

def extract_fbank_features(waveform: torch.FloatTensor, sample_rate: int, output_path: Optional[Path] = None,
                            n_mel_bins: int = 80, overwrite: bool = False):

    if output_path is not None and output_path.is_file() and not overwrite:
        return

    _waveform, _ = convert_waveform(waveform, sample_rate, to_mono=True)
    # Kaldi compliance: 16-bit signed integers
    _waveform = _waveform * (2 ** 15)
    _waveform = _waveform.numpy()

    features = _get_torchaudio_fbank(_waveform, sample_rate, n_mel_bins)
    if features is None:
        raise ImportError(
            "Please install pyKaldi or torchaudio to enable fbank feature extraction"
        )

    if output_path is not None:
        np.save(output_path.as_posix(), features)

    return features

def convert_waveform(waveform: Union[np.ndarray, torch.Tensor], sample_rate: int, normalize_volume: bool = False,
                    to_mono: bool = False, to_sample_rate: Optional[int] = None) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    """convert a waveform:
    - to a target sample rate
    - from multi-channel to mono channel
    - volume normalization

    Args:
        waveform (numpy.ndarray or torch.Tensor): 2D original waveform
            (channels x length)
        sample_rate (int): original sample rate
        normalize_volume (bool): perform volume normalization
        to_mono (bool): convert to mono channel if having multiple channels
        to_sample_rate (Optional[int]): target sample rate
    Returns:
        waveform (numpy.ndarray): converted 2D waveform (channels x length)
        sample_rate (float): target sample rate
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
        _waveform = torch.from_numpy(waveform) if is_np_input else waveform
        converted, converted_sample_rate = ta_sox.apply_effects_tensor(
            _waveform, sample_rate, effects
        )
        if is_np_input:
            converted = converted.numpy()
        return converted, converted_sample_rate
    return waveform, sample_rate

def _get_torchaudio_fbank(waveform: np.ndarray, sample_rate, n_bins=80) -> Optional[np.ndarray]:
    """Get mel-filter bank features via TorchAudio."""
    import torchaudio.compliance.kaldi as ta_kaldi

    waveform = torch.from_numpy(waveform)
    features = ta_kaldi.fbank(
        waveform, num_mel_bins=n_bins, sample_frequency=sample_rate
    )
    return features.numpy()

def filter_manifest_df(
    df, is_train_split=False, extra_filters=None, min_n_frames=5, max_n_frames=3000
):
    filters = {"no speech": df["audio"] == "",
                f"short speech (<{min_n_frames} frames)": df["n_frames"] < min_n_frames,
                "empty sentence": df["tgt_text"] == ""
    }

    if is_train_split:
        filters[f"long speech (>{max_n_frames} frames)"] = df["n_frames"] > max_n_frames
    if extra_filters is not None:
        filters.update(extra_filters)

    invalid = reduce(lambda x, y: x | y, filters.values())
    valid = ~invalid
    print(
        "| "
        + ", ".join(f"{n}: {f.sum()}" for n, f in filters.items())
        + f", total {invalid.sum()} filtered, {valid.sum()} remained."
    )
    return df[valid]

def save_df_to_tsv(dataframe, path: Union[str, Path]):
    _path = path if isinstance(path, str) else path.as_posix()
    dataframe.to_csv(
        _path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )

def create_zip(data_root: Path, zip_path: Path):
    paths = list(data_root.glob("*.npy"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as f:
        for path in tqdm(paths):
            f.write(path, arcname=path.name)

def get_zip_manifest(zip_path: Path, zip_root: Optional[Path] = None):
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

def get_fbank_wave_from_zip(zip_path: Path, manifest_path: Path, zip_root: Optional[Path] = None):
    '''
    Return the fbank from the zip in numpy format
    '''
    _zip_path = Path.joinpath(zip_root or Path(""), zip_path)
    _path,  offset, file_size = str(manifest_path).split(':')
    offset, file_size = int(offset), int(file_size)
    
    with open(_zip_path, "rb") as f:
        f.seek(offset)
        byte_data = f.read(file_size)
        byte_data_fp = io.BytesIO(byte_data)
        fbank_wave = np.load(byte_data_fp)
    return fbank_wave

def is_npy_data(data: bytes) -> bool:
    return data[0] == 147 and data[1] == 78
