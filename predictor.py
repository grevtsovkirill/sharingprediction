import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

seed=8
np.random.seed(seed)



import argparse
parser = argparse.ArgumentParser(description='Prepare classifier')
parser.add_argument('-t','--type', required=True, type=str, choices=['plot', 'train','read','apply'], help='Choose processing type: explore variable [plot], train the model [train], load previously trained model to do plots [read] or apply existing model [apply] ')

args = parser.parse_args()

process_type = vars(args)["type"]


class HistoricalData:
    def __init__(self, data_dir, filename):
        self.data_dir=data_dir
        self.filename=filename
        
    def read_all_data(self):
        '''
        Read all available data
        '''
        dataset=pd.read_csv(os.path.join(self.data_dir, self.filename))
        return dataset

def main():
    data_path = '../Bike-Sharing-Dataset/'
    filename = 'hour.csv'
    histdata = HistoricalData(data_path,filename)
    ds = histdata.read_all_data()
    if process_type=='plot':
        print(ds.head())


if __name__ == "__main__":
    main()

    
