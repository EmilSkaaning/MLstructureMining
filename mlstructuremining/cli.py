import argparse
import datetime
import os
from typing import Optional, List

import numpy as np
import pandas as pd

from mlstructuremining.data_module import DataLoader
from mlstructuremining.utils import load_model, show_best


_BANNER = """
This is a package which takes a directory of PDF files
or a specific file. It then determines the best structural
candidates based off a range of crystallographic structures.
"""

def create_parser() -> argparse.ArgumentParser:
    """
    Create and return the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(prog='ciff',
                                     description=_BANNER,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Load data args
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument("-d", "--data", required=True, type=str,
                                help="a directory of PDFs or a file.")
    parser.add_argument("-n", "--n_cpu", default=1, type=int,
                        help="Number of cpus used by model")
    parser.add_argument("-s", "--show", default=5, type=int,
                        help="Number of best predictions printed")
    parser.add_argument("-f", "--file_name", default='', type=str,
                        help="Name of the output file")

    return parser


def main(args: Optional[List[str]] = None) -> None:
    """
    Main function to run the package functionality.

    Parameters
    ----------
    args : List[str], optional
        List of arguments to parse. If None, uses sys.argv.
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args=args)

    # Constants
    droplist = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Uiso', 'Psize', 'rmin', 'rmax', 'rstep', 'qmin',
                'qmax', 'qdamp', 'delta2']

    # Load data
    data_obj = DataLoader(parsed_args.data)

    current_time = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')
    root_dir = os.getcwd()
    new_dir_path = os.path.join(root_dir, f'ciff_ress_{current_time}')
    os.mkdir(new_dir_path)
    print(f'\n{new_dir_path} has been created!')

    # Make predictions
    bst, df_stru_catalog = load_model(parsed_args.n_cpu)
    for r, gr, gr_XGB, stem_name in data_obj:
        print(f'\n{stem_name}')
        pred = bst.predict(gr_XGB)
        best_list = np.argsort(pred)
        show_best(pred[0], best_list[0], df_stru_catalog, parsed_args.show)

        # Output results
        df_dict = {
            'label': df_stru_catalog['Label'].values,
            'probability': pred[0],
            'similar': df_stru_catalog['Similar'].values
        }
        df = pd.DataFrame(data=df_dict).sort_values('probability', ascending=False).reset_index(drop=True)
        df.to_csv(os.path.join(new_dir_path, f'{stem_name}.csv'))
