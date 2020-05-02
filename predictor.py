import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
import xgboost
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
seed=8
np.random.seed(seed)

import seaborn as sns


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

    def make_full_date(self,df):
        df.loc[:,'fulldteday'] = df.loc[:,'dteday']
        #df.loc[:,'fulldteday'] = 
        return df

    def split_train_test(self):
        df_all = self.read_all_data()
        
        #def prep_ds(self):
        
class ModelDataPrep:
    def __init__(self, df, varlist, target='cnt', splittype='year'):
        self.df = df
        self.varlist = varlist
        self.target = target        
        self.splittype = splittype

        xx = self.df[self.varlist]
        xsc = StandardScaler().fit(xx)
        self.xsc = xsc
        yy = self.df[self.target]
        yy = np.array(yy) 
        ysc = StandardScaler().fit(yy.reshape(-1, 1))
        self.ysc = ysc


    def gen_sample(self, d ):
        X = d[self.varlist]
        Y = d[self.target]
        X = self.xsc.transform(X)
        Y = np.array(Y)
        Y = np.squeeze(self.ysc.transform(Y.reshape(-1, 1)))
        return X,Y
    
    def set_split(self):
        if self.splittype == 'year':
            train = self.df.loc[self.df.yr==0]
            test = self.df.loc[self.df.yr==1]
            x_train,y_train = self.gen_sample(train)            
            x_test,y_test = self.gen_sample(test)
            
        self.X_train = x_train
        self.Y_train = y_train
        self.X_test = x_test
        self.Y_test = y_test

def plot_all(data):
    varlist = data.select_dtypes(include='number').columns
    print("N_num_cols=",len(varlist))
    for i in varlist:
        plt.figure(i) 
        sns.distplot(data[i])
        if i in ['season','yr','mnth','hr','holiday','weekday','workingday','weathersit']:
            print(i,' = ',len(data[i].unique()))
            sns.distplot(data[i],bins=len(data[i].unique()), kde=False)
        #plt.plot(data[i])
        #plt.hist(data[i])
        plt.savefig("Plots/raw/"+i+".png", transparent=True)
        plt.close(i) 
    
    #print(varlist)
    
def plot_over_time(data,var,time='hr',lab_str=''):
    df1 = data.sort_values(time, ascending=True)
    lab_str = lab_str
    fig, ax = plt.subplots()
    for i in var:
        ax.plot(df1[time], df1[i],label=lab_str)
    fig.autofmt_xdate()
    
    plt.savefig("Plots/tmp.png", transparent=True)

def plot_per_season(ds1):
    s={}        
    for i in range(1,5):
        s[i] = (ds1.loc[ds1.season==i].ncnt, ds1.loc[ds1.season==i].temp)
        
    data = (s[1],s[2],s[3],s[4])
    colors = ("blue", "green", "red","orange")
    groups = ("winter", "spring", "summer","autumn")
    plt.figure("response")        
    for data, color, group in zip(data, colors, groups):
        x, y = data
        plt.scatter(x,y, c=color, label=group,  alpha=0.5, s=30)
    plt.legend(loc=2)
    #plt.show()    
    plt.savefig("Plots/season.png", transparent=True)
    plt.close("response")


def cor_plot(ds,variablelist_all):
    correlations = ds[variablelist_all].corr()
    # plot correlation matrix
    plt.figure("cor") 
    fig = plt.figure(figsize=(25, 25))
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    plt.xticks(rotation=90)
    ax.tick_params(labelbottom=True, labelright=True)
    plt.xticks(rotation=90)
    ticks = np.arange(0,len(variablelist_all),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(variablelist_all)
    ax.set_yticklabels(variablelist_all)
    plt.savefig("Plots/cor_all_variabls.png", transparent=True)
    plt.close("cor") 

def scat_plot(ds,variablelist):
    plt.figure("scat") 
    sc = scatter_matrix(ds[variablelist])
    name = ''
    for i in variablelist:
        name +=i+'_'
    plt.savefig("Plots/scat_"+name+".png", transparent=True)
    plt.close("scat") 


def build_model(X,y):
  model = xgboost.XGBRegressor()
  # best_pars = GridSearchCV(model, {"colsample_bytree":[1.0],"min_child_weight":[1.0,1.2]
  #                               ,'max_depth': [3,4,6], 'n_estimators': [500,1000]}, verbose=1)
  # best_pars.fit(X,y)
  # model = xgboost.XGBRegressor(**best_pars.best_params_)
  return model        
    
    
        
def main():
    data_path = '../Bike-Sharing-Dataset/'
    filename = 'hour.csv'
    histdata = HistoricalData(data_path,filename)
    ds = histdata.read_all_data()
    ds = histdata.make_full_date(ds)
    if process_type=='plot':
        print(ds[:25])
        ds1 = ds.copy() #
        max_cnt_val = ds1.cnt.max()
        ds1.loc[:,'ncnt'] = ds1.loc[:,'cnt']/max_cnt_val
        # plot_over_time(ds1,['ncnt', 'temp'],'instant') #
        # plot_per_season(ds1)
        # list_cor = ['hr','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','casual','registered' ,'cnt']        
        # cor_plot(ds1,list_cor)
        # scat_plot(ds1, ['hr','temp','weekday'])
        plot_all(ds1)

    if process_type=='train':
        varlist = ['hr','weekday','weathersit','temp','hum','windspeed']
        data = ModelDataPrep(ds,varlist)
        data.set_split()

        print(data.Y_test[:5])
        model = build_model(data.X_train, data.Y_train)
        model.fit(data.X_train, data.Y_train)
        Y_pred = model.predict(data.X_test)
        print(Y_pred[:5],data.Y_test[:5])
        print(model.score(data.X_test,data.Y_test))
if __name__ == "__main__":
    main()
