import pandas as pd
from typing import Tuple, Dict, List, Optional  #


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    pass


# Example for using typing, very convenient for IntelliSense
def foo(dfs: List[pd.DataFrame]) -> Optional[Dict[int, pd.DataFrame]]:
    result = {}
    i = 0
    for df in dfs:
        result[i] = df
    return result
