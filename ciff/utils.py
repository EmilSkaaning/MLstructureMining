import ast
import time
import pkg_resources
import pandas as pd
import xgboost as xgb
import numpy as np
from typing import Tuple, List



def load_model(n_cpu: int) -> Tuple[xgb.Booster, pd.DataFrame]:
    """
    Load a pre-trained XGBoost model from the package's resources.

    Parameters
    ----------
    n_cpu : int
        Number of CPU threads to use for the XGBoost model.

    Returns
    -------
    tuple
        - xgb.Booster: The loaded XGBoost model.
        - pd.DataFrame: The structure catalog associated with the model.
    """
    model_path = 'model'
    start_time = time.time()
    load_files = pkg_resources.resource_listdir(__name__, model_path)

    this_model = next(file for file in load_files if file == 'xgb_model_bayse_optimization_00000.bin')
    stru_catalog = next(file for file in load_files if 'labels' in file)

    stream = pkg_resources.resource_stream(__name__, f'{model_path}/{stru_catalog}')
    df_stru_catalog = pd.read_csv(stream, index_col=0)

    print(f'\nLoading: {this_model}')

    bst = xgb.Booster({'nthread': n_cpu})
    model_path = pkg_resources.resource_filename(__name__, f'{model_path}/{this_model}')
    bst.load_model(model_path)

    end_time = time.time()
    print(f'Took {end_time-start_time:.1f} sec to load model')
    return bst, df_stru_catalog

def show_best(pred: np.ndarray, 
              best_list: np.ndarray, 
              df_stru_catalog: pd.DataFrame, 
              num_show: int) -> None:
    """
    Display the best predictions based on the model output.

    Parameters
    ----------
    pred : np.ndarray
        Predictions from the model.
    best_list : np.ndarray
        List of best predictions.
    df_stru_catalog : pd.DataFrame
        The structure catalog associated with the model.
    num_show : int
        Number of top predictions to show.

    Returns
    -------
    None
    """
    for count, idx in enumerate(reversed(best_list[-num_show:])):
        print('{}) File: {}, prob: {:3.1f}%'.format(count, df_stru_catalog.iloc[idx]["Label"], pred[idx]*100))
        if not pd.isna(df_stru_catalog.at[idx, "Similar"]):
            similar_files = extract_filenames(df_stru_catalog.at[idx, "Similar"])
            print("    Similar structure COD-IDs:", *similar_files)

def extract_filenames(file_string: str) -> List[str]:
    """
    Extract filenames from a string representation of a list without the .csv extension.

    Parameters
    ----------
    file_string : str
        String representation of a list of filenames with .csv extension.

    Returns
    -------
    list[str]
        List of filenames without the .csv extension.
    """
    file_list = ast.literal_eval(file_string)

    return [filename[:-4] for filename in file_list]