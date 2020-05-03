from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


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
