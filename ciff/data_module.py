import numpy as np
import pandas as pd
import os
import xgboost as xgb
from scipy import interpolate
import warnings


class data_loader():
    def __init__(self, data_dir):
        if os.path.isfile(data_dir):  # is it a file
            print('Input is file')
            r, gr = self.load_data_set(data_dir)
            r, gr = self.interpolate_pdf(r, gr)
            self.r, self.gr, self.gr_XGB, self.data_name = [r], [gr], [xgb.DMatrix([gr.T])], [data_dir]

        elif os.path.isdir(data_dir):  # is it a directory
            print('Input is directory')
            files = os.listdir(data_dir)
            files = [file for file in files if os.path.isfile(f'{data_dir}/{file}') and file[0] not in ["_", "."]]
            self.r, self.gr, self.gr_XGB, self.data_name = [], [], [], []
            for file in files:
                r, gr = self.load_data_set(f'{data_dir}/{file}')
                r, gr = self.interpolate_pdf(r, gr)
                self.r.append(r), self.gr.append(gr), self.gr_XGB.append(xgb.DMatrix([gr.T])), self.data_name.append(file)


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

    def load_data_set(self, file_path):
        _SKIP_HEADER_TRIES = 100

        for skiprows in range(_SKIP_HEADER_TRIES):
            try:
                data = np.loadtxt(file_path, skiprows=skiprows).T
                break
            except ValueError:
                pass
        else:
            warnings.warn("Failed to load data after trying to skip headers for {} times.".format(_SKIP_HEADER_TRIES))
            raise Exception("Data loading failed due to incompatible format or too many header lines.")
        
        r, gr = data[0], data[1]
        self.check_array_values(arr=r)
        gr /= np.amax(gr)

        return r, gr

    def interpolate_pdf(self, x, y):
        f = interpolate.interp1d(x, y)
        x_new = np.arange(0, 30.1, 0.1)
        y_new = f(x_new)
        return x_new, y_new

    def check_array_values(self, arr: np.ndarray, min_val: int = 0, max_val: int = 30) -> bool:
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

        Warnings
        --------
        If the array does not meet the criteria, a warning is issued indicating the expected 
        criteria based on provided `min_val` and `max_val`.

        Examples
        --------
        >>> arr = np.array([0, 5, 10, 15, 20, 25, 30, 35])
        >>> check_array_values(arr)
        True

        >>> arr = np.array([5, 10, 15, 20, 25])
        >>> check_array_values(arr, min_val=5, max_val=50)
        False
        """

        if np.min(arr) != min_val or np.max(arr) < max_val:
            warnings.warn(f"Array does not meet the criteria: Minimum value should be {min_val} and maximum value should be at least {max_val}.")
            raise Exception(f"Array does not meet the criteria: Minimum value should be {min_val} and maximum value should be at least {max_val}.")
        return True