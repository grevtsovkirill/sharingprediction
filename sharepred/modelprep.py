from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
import numpy as np
import xgboost
import pickle

seed=8
np.random.seed(seed)
        
class ModelDataPrep:
    def __init__(self, df, varlist, target='cnt', splittype='year',test_samp_size=0.33):
        self.df = df
        self.varlist = varlist
        self.target = target        
        self.splittype = splittype
        self.test_samp_size = test_samp_size

    def gen_sample(self, d ):
        X = d[self.varlist]
        Y = d[self.target]
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
        with open('model_xgb_best.pickle', 'wb') as f:
            pickle.dump(model, f)
    elif opt=='load':
        with open('model_xgb_best.pickle', 'rb') as f:
            model = pickle.load(f)
        
    return model        
    
    
