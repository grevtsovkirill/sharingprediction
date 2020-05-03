import pandas as pd
import numpy as np
import os

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

    def convert_season(self,df):
        df = pd.concat([df,pd.get_dummies(df['season'], prefix='season')],axis=1)
        return df
        #def prep_ds(self):

    def convert_hr(self,df):
        df = pd.concat([df,pd.get_dummies(df['hr'], prefix='hr')],axis=1)
        return df

    def convert_mnth(self,df):
        df = pd.concat([df,pd.get_dummies(df['mnth'], prefix='mnth')],axis=1)
        return df

    def convert_weekday(self,df):
        df = pd.concat([df,pd.get_dummies(df['weekday'], prefix='weekday')],axis=1)
        return df
    
    def transform_target(self,df):
        df.loc[:,'ncnt'] = np.log(df.loc[:,'cnt'])
        return df  
