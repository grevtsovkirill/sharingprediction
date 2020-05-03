import sharepred as sp

import argparse
parser = argparse.ArgumentParser(description='Prepare regression for bike sharing prediction')
parser.add_argument('-t','--type', required=True, type=str, choices=['plot', 'train','read','apply'], help='Choose processing type: explore variable [plot], train the model [train], load previously trained model to do plots [read] or apply existing model [apply] ')

args = parser.parse_args()

process_type = vars(args)["type"]


        
def main():
    data_path = '../Bike-Sharing-Dataset/'
    filename = 'hour.csv'
    histdata = sp.dataprep.HistoricalData(data_path,filename)
    ds = histdata.read_all_data()
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
        sp.weekday_feature(ds1,'ncnt','season')
        sp.day_hour_feature(ds1,'ncnt',1)
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
        to_del = ['dteday','season','mnth','instant','dteda','atemp','weekday','holiday','cnt','ncnt','casual','registered']
        varlist = list(set(full_list)-set(to_del))
        print("final list: ", varlist)
        data = sp.ModelDataPrep(ds,varlist,'ncnt','null')
        data.set_split()

        print(data.Y_test[:5])
        model = sp.build_model(data.X_train, data.Y_train,'load')
        model.fit(data.X_train, data.Y_train)
        Y_pred = model.predict(data.X_test)
        print(Y_pred[:5],data.Y_test[:5])
        print(model.score(data.X_test,data.Y_test))

if __name__ == "__main__":
    main()
