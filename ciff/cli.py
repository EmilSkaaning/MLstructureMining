import argparse, os, datetime
import numpy as np
import pandas as pd
from ciff.data_module import DataLoader
from ciff.utils import load_model, show_best


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

parser.add_argument("-n", "--n_cpu", default=1, type=int,
                    help="Number of cpus used by model")

parser.add_argument("-s", "--show", default=5, type=int,
                    help="Number of best predictions printed")

parser.add_argument("-f", "--file_name", default='', type=str,
                    help="Name of the output file")

def main(args=None):
    args = parser.parse_args(args=args)
    droplist = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Uiso', 'Psize', 'rmin', 'rmax', 'rstep', 'qmin',
                'qmax', 'qdamp', 'delta2']

    # load data
    data_obj = DataLoader(args.data)

    ct = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')
    root = os.getcwd()
    new_dir = os.path.join(root, f'ciff_ress_{ct}')
    os.mkdir(f'{new_dir}')
    print(f'\n{new_dir} has been created!')

    # do predictions
    bst, df_stru_catalog = load_model(args.n_cpu)
    for r, gr, gr_XGB, stem_name in data_obj:
        print(f'\n{stem_name}')
        pred = bst.predict(gr_XGB)
        best_list = np.argsort(pred)
        show_best(pred[0], best_list[0], df_stru_catalog, args.show)

        # outputs
        df_dict = {
            'label': df_stru_catalog['Label'].values,
            'probability': pred[0],
            'similar': df_stru_catalog['Similar'].values
        }
        df = pd.DataFrame(data=df_dict)
        df = df.sort_values('probability', ascending=False)
        df = df.reset_index(drop=True)

        df.to_csv(os.path.join(new_dir, f'{stem_name}.csv'))


