import numpy as np
import pandas as pd
import pickle
import sys

import helpers
import split_helpers

def main():
    helpers.check_args(sys.argv, 3)
    read_fn = sys.argv[1].split("=")[1]
    write_dir = sys.argv[2].split("=")[1]
    split_helpers.split_dataset(read_fn, write_dir)

if __name__ == "__main__":
    main()