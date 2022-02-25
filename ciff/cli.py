import argparse, os, sys, time, pkg_resources, h5py
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
"""
print(__name__)
stream = pkg_resources.resource_stream(__name__, 'model/small/answers_metal_S_H_1_11_AA_split_00100_w_abc_uiso.csv')
df = pd.read_csv(stream, index_col=0)
print(df.head())
print(pkg_resources.resource_listdir(__name__, 'model/small/data'))
"""

_BANNER = """
This is a package which takes a directory of PDF files
or a specific file. It then determines the best structural
candidates based of a metal oxide catalog. Results can
be compared with precomputed PDF through Pearson analysis.
"""

parser = argparse.ArgumentParser(prog='ciff',
                        description=_BANNER, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Load data args
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument("-d", "--data", default=None, type=str,
                    help="a directory of PDFs or a file.", required=True)

# model args
parser.add_argument("-m", "--model", default=0, type=int,
                    choices=[0, 1, 2],
                    help="Choose what model to load. 0 large, 1 small and 2 both")

parser.add_argument("-n", "--nthreads", default=1, type=int,
                    help="Number of threads used by model")

parser.add_argument("-s", "--show", default=5, type=int,
                    help="Number of best predictions printed")

# Pearson args
parser.add_argument("-p", "--pearson", default=5, type=int,
                    help="Calculate the Pearson correlation coefficient from pre-calculated PDFs of best suggested models.")

# output args
parser.add_argument("-o", "--output", default=True, type=bool,
                    help="Save a .csv with results")

parser.add_argument("-f", "--file_name", default='', type=str,
                    help="Name of the output file")

parser.add_argument("-P", "--plot", default=True, type=bool,
                    help="Plot 5 best prediction")

def main(args=None):
    args = parser.parse_args(args=args)
    droplist = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Uiso', 'Psize', 'rmin', 'rmax', 'rstep', 'qmin',
                'qmax', 'qdamp', 'delta2']

    # load data
    data_obj = data_loader(args.data, args.model)
    if data_obj.is_file:  # todo: implement for dir
        stem_name = f'{args.file_name}_{data_obj.data_name}_{args.model}'

    # do predictions
    bst, answer_list, df_stru_catalog = load_model(args.model, args.nthreads)
    pred = bst.predict(data_obj.gr_XGB)
    best_list = np.argsort(pred)
    show_best(pred, best_list, answer_list, df_stru_catalog, args.show)

    # get Pearson values
    if args.pearson != 0:
        print('\nCalculating Pearson Correlation Coeficients')
        if args.pearson > len(answer_list) or args.pearson == -1:
            pearson_list = pearson_correlation(data_obj.r, data_obj.gr, args.model, best_list, droplist, len(answer_list), stem_name, args.plot)
        else:
            pearson_list = pearson_correlation(data_obj.r, data_obj.gr, args.model, best_list, droplist, args.pearson, stem_name, args.plot)

    # outputs
    if args.output:
        print('\nGenerating outputs')
        df_dict = {
            'label': answer_list,
            'probability': pred[0],
            'similar': df_stru_catalog['Similar'].values
        }
        df = pd.DataFrame(data=df_dict)
        df = df.sort_values('probability', ascending=False)
        df = df.reset_index(drop=True)
        if args.pearson != 0:
            df['pearson'] = pearson_list

        df.to_csv(f'{stem_name}.csv')
        print(df.head())


def plot_best(r, grs, pears, labels, file_name):
    FIGSIZE = (14, 16)
    LABEL_FONTSIZE = 30
    LEGEND_SIZE = 20
    TICK_LABEL = 20
    TICK_LENGTH = 7
    TICK_WIDTH = 2
    LINEWIDTH = 5

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for idx, (gr, pear, pdf_label) in enumerate(zip(grs, pears, labels)):
        plt.plot(r, gr-idx, label=f'{pdf_label} - {pear:.2f}', linewidth=LINEWIDTH)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=LEGEND_SIZE)
    plt.ylabel('G(r) / a.u.', fontsize=LABEL_FONTSIZE)
    plt.xlabel(u'r / Å', fontsize=LABEL_FONTSIZE)
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,
        labelleft=False)  # ticks along the bottom edge are off
    ax.tick_params(length=TICK_LENGTH, width=TICK_WIDTH, labelsize=TICK_LABEL)
    plt.xlim(r[0], r[-1])
    plt.tight_layout()
    plt.savefig(f'{file_name}_plot.png', dpi=300)
    return None


def pearson_correlation(r, gr, model_path, best_list, droplist, num_pears, file_name, do_plot):
    if model_path == 0:
        data_path = 'model/large/large_data.hdf5'
    else:
        data_path = 'model/small/small_data.hdf5'
    #data_files = sorted(pkg_resources.resource_listdir(__name__, data_path))
    data_file = pkg_resources.resource_stream(__name__, data_path)
    f = h5py.File(data_file, "r")
    key_list = list(f.keys())

    best_pear = np.empty(len(best_list[0]))
    best_pear[:] = np.nan
    gr_plot, pear_plot, label_plot = [gr], [1], ['Data']
    pbar = tqdm(total=num_pears)
    for count, idx in enumerate(best_list[0][::-1]):
        H = [x.decode() for x in f['columns']]
        df = pd.DataFrame(f[key_list[idx]], columns=H)
        #stream = pkg_resources.resource_stream(__name__, data_path+'/'+data_files[idx])

        #df = pd.read_csv(stream)

        df = df.drop(droplist, axis=1)
        pcc_list = []
        for row_idx in range(len(df)):
            pcc, _ = stats.pearsonr(df.iloc[row_idx].to_numpy(), gr)
            pcc_list.append(pcc)
        best_pear[count] = np.amax(pcc_list[0])

        if do_plot and count < 5:
            gr_plot.append(df.iloc[np.argmax(pcc_list[0])].to_numpy())
            pear_plot.append(np.amax(pcc_list[0]))
            label_plot.append(key_list[idx])

        pbar.update()
        if count == num_pears -1:
            break
    pbar.close()
    f.close()

    plot_best(r, gr_plot, pear_plot, label_plot, file_name)
    return best_pear


class data_loader():
    def __init__(self, data_dir, which_model):
        self.model = which_model
        if os.path.isfile(data_dir):  # is it a file
            print('Input is file')
            r, gr, gr_XGB = self.load_data_set(data_dir)
            if '/' in data_dir:
                self.data_name = data_dir.rsplit('/',1)[1]
            else:
                self.data_name = data_dir
            self.data_name = self.data_name.rsplit('.', 1)[0]
            self.is_file = True
        elif os.path.isdir(data_dir):  # is it a directory
            print('Input is directory')
            self.is_file = False
            pass
        else:  # this should give an error
            print(data_dir, 'is not valid')
            sys.exit()

        self.r = r
        self.gr = gr
        self.gr_XGB = gr_XGB


    def load_data_set(self, file_path):
        for skiprow in range(50):
            try:
                #stream = pkg_resources.resource_stream(os.getcwd(), file_path)
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


def load_model(model_path, nthreads):
    if model_path == 0:
        model_path = f'model/large'
    else:
        model_path = f'model/small'

    start_time = time.time()
    load_files = pkg_resources.resource_listdir(__name__, model_path)

    this_model = [file for file in load_files if '.bin' in file][0]
    stru_catalog = [file for file in load_files if 'structure_catalog' in file][0]
    answer_sheet = [file for file in load_files if 'answer' in file][0]

    stream = pkg_resources.resource_stream(__name__, f'{model_path}/{stru_catalog}')
    df_stru_catalog = pd.read_csv(stream, index_col=0)

    stream = pkg_resources.resource_stream(__name__, f'{model_path}/{answer_sheet}')
    df_answer = pd.read_csv(stream, index_col=0)
    answer_list = list(df_answer.index)
    print(f'\nLoading: {this_model}')

    bst = xgb.Booster({'nthread': nthreads})  # init model
    #stream = pkg_resources.resource_string(__name__, f'{model_path}/{this_model}')
    #print(stream)
    model_path = pkg_resources.resource_filename(__name__, f'{model_path}/{this_model}')
    #print(model_path)
    #stream = io.TextIOWrapper(model_path)
    bst.load_model(model_path)#io.BytesIO(model_path))  # load data

    end_time = time.time()
    print(f'Took {end_time-start_time:.1f} sec to load model')
    return bst, answer_list, df_stru_catalog

def show_best(pred, best_list, answer_list, df_stru_catalog, num_show):
    for count, idx in enumerate(best_list[0][::-1]):
        print('\n{}) File: {}, prob: {:.4f}'.format(count, answer_list[idx], pred[0][idx]))
        new_list = pretty_read(df_stru_catalog.iloc[idx]["Similar"])
        if len(new_list) > 1:
            print(f'Additional files in structure catalog: {*new_list[1:],}')

        if count == num_show:
            break
    return None


def pretty_read(this_str):
    this_str = this_str.replace("['", '')
    this_str = this_str.replace("']", '')
    this_str = this_str.split("', '")

    return this_str
