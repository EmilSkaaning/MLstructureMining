import os
import sys
import warnings
import numpy as np
import xgboost as xgb
from scipy import interpolate
from typing import Tuple


class DataLoader:
    """
    DataLoader class to load and preprocess data for XGBoost models.
    The class can handle both single files and directories.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the DataLoader with the given data directory or file.

        Parameters
        ----------
        data_dir : str
            Path to the data file or directory.
        """
        if os.path.isfile(data_dir):  # is it a file
            heads, tails = os.path.split(data_dir)
            print('Input is file')
            r, gr = self.load_data_set(data_dir)
            r, gr = self.interpolate_pdf(r, gr)
            self.r, self.gr, self.gr_XGB, self.data_name = [r], [gr], [xgb.DMatrix([gr.T])], [tails.rsplit(".", 1)[0]]

        elif os.path.isdir(data_dir):  # is it a directory
            print('Input is directory')
            files = [file for file in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, file)) and not file.startswith(("_", "."))]
            self.r, self.gr, self.gr_XGB, self.data_name = [], [], [], []
            for file in files:
                r, gr = self.load_data_set(os.path.join(data_dir, file))
                r, gr = self.interpolate_pdf(r, gr)
                self.r.append(r)
                self.gr.append(gr)
                self.gr_XGB.append(xgb.DMatrix([gr.T]))
                self.data_name.append(file.rsplit(".", 1)[0])

        else:  # this should give an error
            print(data_dir, 'is not valid')
            sys.exit()

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        try:
            out_data_name = self.data_name[self.index]
            out_r = self.r[self.index]
            out_gr = self.gr[self.index]
            out_gr_XGB = self.gr_XGB[self.index]
            self.index += 1
        except IndexError:
            raise StopIteration
        return out_r, out_gr, out_gr_XGB, out_data_name

    def load_data_set(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from the given file path.

        Parameters
        ----------
        file_path : str
            Path to the data file.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            r and gr values loaded from the file.
        """
        _SKIP_HEADER_TRIES = 100

        for skiprows in range(_SKIP_HEADER_TRIES):
            try:
                data = np.loadtxt(file_path, skiprows=skiprows).T
                break
            except ValueError:
                pass
        else:
            warnings.warn(f"Failed to load data after trying to skip headers for {_SKIP_HEADER_TRIES} times.")
            raise Exception("Data loading failed due to incompatible format or too many header lines.")
        
        r, gr = data[0], data[1]
        self.check_array_values(arr=r)
        gr /= np.amax(gr)

        return r, gr

    def interpolate_pdf(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate the given x and y values to a new grid.

        Parameters
        ----------
        x : np.ndarray
            Original x values.
        y : np.ndarray
            Original y values.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Interpolated x and y values.
        """
        f = interpolate.interp1d(x, y)
        x_new = np.arange(0, 30.1, 0.1)
        y_new = f(x_new)
        return x_new, y_new

    @staticmethod
    def check_array_values(arr: np.ndarray, min_val: int = 0, max_val: int = 30) -> bool:
        """
        Check if the minimum value of the array equals `min_val` and its maximum value is at least `max_val`.

        Parameters
        ----------
        arr : np.ndarray
            Input numpy array to be checked.
        min_val : int, optional
            The value that the minimum of the array should be. Default is 0.
        max_val : int, optional
            The value that the maximum of the array should be at least. Default is 30.

        Returns
        -------
        bool
            True if the conditions are met, False otherwise.

        Raises
        ------
        Exception
            If the array does not meet the criteria, an exception is raised indicating the expected criteria.

        Examples
        --------
        >>> arr = np.array([0, 5, 10, 15, 20, 25, 30, 35])
        >>> DataLoader.check_array_values(arr)
        True

        >>> arr = np.array([5, 10, 15, 20, 25])
        >>> DataLoader.check_array_values(arr, min_val=5, max_val=50)
        False
        """
        if np.min(arr) != min_val or np.max(arr) < max_val:
            warnings.warn(f"Array does not meet the criteria: Minimum value should be {min_val} and maximum value should be at least {max_val}.")
            raise Exception(f"Array does not meet the criteria: Minimum value should be {min_val} and maximum value should be at least {max_val}.")
        return True
