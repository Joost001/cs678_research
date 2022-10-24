import pandas as pd
from pathlib import Path
from typing import Union

def load_df_from_tsv(path: Union[str, Path]) -> pd.DataFrame: #note that Union[str, Path] is equal to str | Path
    _path = path if isinstance(path, str) else path.as_posix() #as_posix() Return a string representation of the path with forward slashes (/). Ex: p.as_posix()->'c:/windows'
    return pd.read_csv(_path, sep='\t', header=0, encoding='utf-8', escapeachar='\\', quoting=csv.QUOTE_NONE, na_filter=False)
