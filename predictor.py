import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


seed=8
np.random.seed(seed)



import argparse
parser = argparse.ArgumentParser(description='Prepare classifier')
parser.add_argument('-t','--type', required=True, type=str, choices=['plot', 'train','read','apply'], help='Choose processing type: explore variable [plot], train the model [train], load previously trained model to do plots [read] or apply existing model [apply] ')

args = parser.parse_args()

process_type = vars(args)["type"]
