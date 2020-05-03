import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
import xgboost
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
seed=8
np.random.seed(seed)

import seaborn as sns

import dataprep 
import argparse
parser = argparse.ArgumentParser(description='Prepare classifier')
parser.add_argument('-t','--type', required=True, type=str, choices=['plot', 'train','read','apply'], help='Choose processing type: explore variable [plot], train the model [train], load previously trained model to do plots [read] or apply existing model [apply] ')

args = parser.parse_args()

process_type = vars(args)["type"]


        
class ModelDataPrep:
    def __init__(self, df, varlist, target='cnt', splittype='year',test_samp_size=0.33):
        self.df = df
        self.varlist = varlist
        self.target = target        
        self.splittype = splittype
        self.test_samp_size = test_samp_size

        xx = self.df[self.varlist]
        xsc = StandardScaler().fit(xx)
        self.xsc = xsc
        # yy = self.df[self.target]
        # yy = np.array(yy) 
        # ysc = StandardScaler().fit(yy.reshape(-1, 1))
        # self.ysc = ysc


    def gen_sample(self, d ):
        X = d[self.varlist]
        Y = d[self.target]
        #X = self.xsc.transform(X)
        #Y = np.array(Y)
        #Y = np.squeeze(self.ysc.transform(Y.reshape(-1, 1)))
        return X,Y
    
    def set_split(self):
        if self.splittype == 'year':
            train = self.df.loc[self.df.yr==0]
            test = self.df.loc[self.df.yr==1]
            x_train,y_train = self.gen_sample(train)            
            x_test,y_test = self.gen_sample(test)
        else:
            X,Y = self.gen_sample(self.df)
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = self.test_samp_size)
            
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

def plot_per_season(ds1,varlist,hvar="yr"):
    for i in varlist:
        plt.figure(i)        
        sns.catplot(x="season", y=i, hue=hvar,  data=ds1);
        plt.savefig("Plots/season/season_"+hvar+"_"+i+".png", transparent=True)
        plt.close(i)


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

def weekday_feature(ds,var,hvar):
    plt.figure(var)        
    sns.catplot(x="weekday", y=var,   hue=hvar, kind="box",data=ds)
    plt.savefig("Plots/days/weekday_"+var+".png", transparent=True)
    plt.close(var)
    
def day_hour_feature(ds,var,seas):
    plt.figure(var)
    #map_df = ds.loc[ds.season==seas].pivot('weekday','hr',var)
    ordered_days = ds.weekday.value_counts().index
    g = sns.FacetGrid(ds.loc[ds.season==seas], row="weekday") #, row_order=ordered_days, height=1.7, aspect=4)
    g.map(sns.distplot, "hr", hist=True, rug=True);
    #sns.heatmap(map_df)
    plt.savefig("Plots/days/dh_"+var+"_"+str(seas)+".png", transparent=True)
    plt.close(var)
    
    
def test_hr(df):
    val = sum(range(24))
    n_susp = 0
    tot_diff = 0
    for i in df.dteday.unique():
        # if df.hr.loc[df.dteday==i].sum() != val:
        #     print(i,df.hr.loc[df.dteday==i].sum())
        if len(df.hr.loc[df.dteday==i]) != 24 or df.hr.loc[df.dteday==i].sum() != val:
            n_susp+=1
            tot_diff += len(df.hr.loc[df.dteday==i])
            #print(i,len(df.hr.loc[df.dteday==i])," " , df.hr.loc[df.dteday==i].sum())

    print(n_susp, tot_diff)
    
def build_model(X,y,opt='def'):
    model = xgboost.XGBRegressor()

    if opt=='def':
        with open('model_xgb_def.pickle', 'wb') as f:
            pickle.dump(model, f)

    elif opt=='opt':
        best_pars = GridSearchCV(model, {"colsample_bytree":[1.0],"min_child_weight":[1.0,1.2]
                                         ,'max_depth': [3,4,6], 'n_estimators': [500,1000]}, verbose=1)
        best_pars.fit(X,y)        
        model = xgboost.XGBRegressor(**best_pars.best_params_)
        #model.save('model_xgb_best.h5')
        with open('model_xgb_best.pickle', 'wb') as f:
            pickle.dump(model, f)
    elif opt=='load':
        with open('model_xgb_best.pickle', 'rb') as f:
            model = pickle.load(f)
        
    return model        
    
    
        
def main():
    data_path = '../Bike-Sharing-Dataset/'
    filename = 'hour.csv'
    histdata = dataprep.HistoricalData(data_path,filename)
    ds = histdata.read_all_data()
    ds = histdata.make_full_date(ds)
    ds = histdata.convert_season(ds)
    ds = histdata.convert_hr(ds)
    ds = histdata.convert_mnth(ds)
    ds = histdata.convert_weekday(ds)
    ds = histdata.transform_target(ds)
    if process_type=='plot':
        print(ds[:25])
        ds1 = ds.copy() #
        # plot_over_time(ds1,['ncnt', 'temp'],'instant') #
        #for i_var in ['yr','workingday']:
        #    plot_per_season(ds1,['temp','atemp','hum','windspeed','cnt'],i_var)
            
        #test_hr(ds1)
        weekday_feature(ds1,'ncnt','season')
        day_hour_feature(ds1,'ncnt',1)
        # list_cor = ['hr','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','casual','registered' ,'cnt']        
        # cor_plot(ds1,list_cor)
        # scat_plot(ds1, ['hr','temp','weekday'])
        #plot_all(ds1)
        #print(ds1.loc[(ds1.atemp<0.35) & (ds1.yr==1) & (ds1.season==3)])
        print("year", len(ds1.loc[ds1.yr==1])," ",len(ds1.loc[ds1.yr==0]))
        print("workingday: 1 = ",len(ds1.loc[ds1.workingday==1]),", 0 = ",len(ds1.loc[ds1.workingday==0]), ", mon-fr = ",len(ds1.loc[ds1.weekday<5]),", st+sund = ",len(ds1.loc[ds1.weekday>4]), ", hol = ",len(ds1.loc[ds1.holiday==1]),)
        
    if process_type=='train':
        #varlist = ['season_1','season_2','season_3','season_4','hr','weekday','weathersit','temp','hum','windspeed']
        full_list = ds.columns
        print(full_list)
        to_del = ['dteday','fulldteday','season','mnth','instant','dteda','atemp','weekday','holiday','cnt','ncnt','casual','registered']
        varlist = list(set(full_list)-set(to_del))
        print("final list: ", varlist)
        data = ModelDataPrep(ds,varlist,'ncnt','null')
        data.set_split()

        print(data.Y_test[:5])
        model = build_model(data.X_train, data.Y_train,'opt')
        model.fit(data.X_train, data.Y_train)
        Y_pred = model.predict(data.X_test)
        print(Y_pred[:5],data.Y_test[:5])
        print(model.score(data.X_test,data.Y_test))
if __name__ == "__main__":
    main()
