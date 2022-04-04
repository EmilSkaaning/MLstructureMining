import xgboost as xgb
import time, pkg_resources, h5py
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from tqdm import tqdm


def pearson_correlation(r, gr, model_path, best_list, droplist, num_pears, file_name, do_plot):
    if model_path == 0:
        data_path = 'model/large/large_data.hdf5'
    else:
        data_path = 'model/small/small_data.hdf5'
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