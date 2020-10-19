import numpy as np
import pandas as pd 
from utils import *

if __name__ == "__main__":
    from os.path import join, dirname, abspath
    # managing paths
    BASE_DIR = dirname(abspath("__file__"))
    DATA_DIR = join(BASE_DIR, "data")
    DATA_PATH = join(DATA_DIR, "weatherHistory.csv")
    
    # reading data
    df = pd.read_csv(DATA_PATH)
    
    """
    # checking data
    basic_data_check(df)
    """

    # cleaning data
    clean_df = data_clean(df)
    
    # basic_data_check(clean_df)

    clean_df.to_csv(join(DATA_DIR, "clean_data.csv"), index=False)
    
