import numpy as np
import pandas as pd
import os
import xgboost as xgb


class data_loader():
    def __init__(self, data_dir, which_model):
        self.model = which_model
        if os.path.isfile(data_dir):  # is it a file
            print('Input is file')
            r, gr, gr_XGB = self.load_data_set(data_dir)
            data_name_ph = self.get_str(data_dir)
            r = [r]
            gr = [gr]
            gr_XGB = [gr_XGB]
            self.data_name = [data_name_ph]

            self.is_file = True

        elif os.path.isdir(data_dir):  # is it a directory
            print('Input is directory')
            files = os.listdir(data_dir)
            files = [file for file in files if os.path.isfile(f'{data_dir}/{file}')]
            r, gr, gr_XGB, self.data_name = [], [], [], []
            for file in files:
                r_ph, gr_ph, gr_XGB_ph = self.load_data_set(f'{data_dir}/{file}')
                data_name_ph = self.get_str(file)
                r.append(r_ph)
                gr.append(gr_ph)
                gr_XGB.append(gr_XGB_ph)
                self.data_name.append(data_name_ph)
            self.is_file = False

        else:  # this should give an error
            print(data_dir, 'is not valid')
            sys.exit()

        self.r = r
        self.gr = gr
        self.gr_XGB = gr_XGB

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

    def get_str(self, data_dir):
        if '/' in data_dir:
            data_name = data_dir.rsplit('/', 1)[1]
        else:
            data_name = data_dir
        data_name = data_name.rsplit('.', 1)[0]

        return data_name

    def load_data_set(self, file_path):
        for skiprow in range(50):
            try:
                data = np.loadtxt(file_path, skiprows=skiprow)
                break
            except ValueError:
                if skiprow == 49:
                    print(f'Could not load {file_path}')
                    raise
                else:
                    continue


        data = data.T
        if data[0][0] == 0.:
            data[1][:] -= data[1][0]
        else:  # do padding
            pass

        if data[0][-1] >= 30.:
            pass
        else:  # do padding
            print('data need minimum rmax of 30')
            sys.exit()

        mean_step = np.mean(data[0][1:]-data[0][:-1])
        if not round(0.1 % mean_step, 3) == 0:
            print('fix data')
            sys.exit()
        else:
            step = round(0.1 / mean_step, 3)

        if self.model == 1:
            # small model
            gr = np.array([val for i, val in enumerate(data[1]) if i % step == 0 and i <= 1101 and i >= 100])
            r = [val for i, val in enumerate(data[0]) if i % step == 0 and i <= 1101 and i >= 100]
        else:
            # large model
            gr = np.array([val for i, val in enumerate(data[1]) if i % step == 0 and i <= 3001])
            r = [val for i, val in enumerate(data[0]) if i % step == 0 and i <= 3001]

        gr /= np.amax(gr)

        df = pd.DataFrame(gr).transpose()
        XGB_mat = xgb.DMatrix(df)

        return r, gr, XGB_mat

