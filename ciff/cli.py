import argparse, os, datetime
import numpy as np
import pandas as pd
from ciff.data_module import data_loader
from ciff.utils import load_model, show_best, pearson_correlation


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

    if data_obj.is_file:
        new_dir = ''
    else:
        ct = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')
        new_dir = f'./ciff_ress_{ct}/'
        os.mkdir(f'{new_dir}')
        print(f'\n{new_dir} has been created!')

    # do predictions
    bst, answer_list, df_stru_catalog = load_model(args.model, args.nthreads)
    for r, gr, gr_XGB, stem_name in data_obj:
        pred = bst.predict(gr_XGB)
        best_list = np.argsort(pred)
        show_best(pred, best_list, answer_list, df_stru_catalog, args.show)

        # get Pearson values
        if args.pearson != 0:
            print('\nCalculating Pearson Correlation Coeficients')
            if args.pearson > len(answer_list) or args.pearson == -1:
                pearson_list = pearson_correlation(r, gr, args.model, best_list, droplist, len(answer_list), f'{new_dir}{stem_name}', args.plot)
            else:
                pearson_list = pearson_correlation(r, gr, args.model, best_list, droplist, args.pearson, f'{new_dir}{stem_name}', args.plot)

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

            df.to_csv(f'{new_dir}{stem_name}.csv')
            print(df.head())


