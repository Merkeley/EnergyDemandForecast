'''

    File: EF_tools.py
        Electicity Demand Forecasting tools for model evaluation and data analysis.
        
    Author : MB
    
    Date Created: 4/7/2020
    Date Modified:
    
    Update Log:
    
    Functions Defined:
        test_stationarity
        order_sweep
        top_results

'''
from statsmodels.tsa.stattools import adfuller
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.tsa.statespace.mlemodel
from fbprophet import Prophet

def agg_weekly(data_frame) :
    '''
        Function: agg_weekly

        Description: Aggregate the energy demand data on a weekly basis.  
        Accumulate the energy demand into a daily total.
        Record the high temperature for each of the cities in the dataset.  
        Then create and return the new dataframe.

        Arguments: data_frame - Pandas dataframe containing the energy demand
            data on an hourly basis.

        Return: Aggregated dataframe with daily demand total and daily high temps.

    '''

    # Aggregate data on a weekly basis
    current_day = data_frame.index[0].date()
    row = data_frame.index[0]
    current_dow = data_frame.loc[row, 'DOW']
    week_demand = 0.0
    X_by_week = pd.DataFrame([], index=[current_day], \
        columns=['week_demand', 'CA_pop', 'la_high', 'sd_high', 'sj_high', 'sf_high',\
                 'sac_high', 'f_high', 'DOW' ])

    # Initialize the variables used to track high temperatures
    la_high = 0
    sd_high = 0
    f_high = 0
    sj_high = 0
    sf_high = 0
    sac_high = 0
    pop=0
    found_tue = False


    # For each row in the dataframe
    for row in data_frame.index :

        # Check the day of the week
        next_dow = data_frame.loc[row].DOW
        if next_dow == 0 and found_tue == False :
            current_day = row.date()

        if next_dow == 1 :
            found_tue = 1

        # If we've transitioned to the beginning of the next week
        if next_dow == 0 and found_tue :
            # Create an entry for the previous week
            X_by_week.loc[current_day] = [week_demand, pop, \
                la_high, sd_high, sj_high, sf_high, sac_high, f_high, current_day.weekday()]
            '''
            {'day_demand' :day_demand, 'la_high': la_high,\
                'sd_high' : sd_high, 'f_high' : f_high, 'DOW' : current_day.weekday()}
            '''
            current_dow = next_dow
            found_tue = False
            la_high = data_frame.loc[row, 'Los Angeles']
            sd_high = data_frame.loc[row, 'San Diego']
            f_high = data_frame.loc[row, 'Fresno']
            sj_high = data_frame.loc[row, 'San Jose']
            sf_high = data_frame.loc[row, 'San Francisco']
            sac_high = data_frame.loc[row, 'Sacramento']
            pop=0
            week_demand = data_frame.loc[row, 'demand']
        else :
            # If we are still working on the weekly values then adjust as needed.
            week_demand += data_frame.loc[row, 'demand']
            pop = data_frame.loc[row, 'CA_pop']
            if data_frame.loc[row, 'Los Angeles'] > la_high :
                la_high = data_frame.loc[row, 'Los Angeles']
            if data_frame.loc[row, 'San Diego'] > sd_high :
                sd_high = data_frame.loc[row, 'San Diego']
            if data_frame.loc[row, 'Fresno'] > f_high :
                f_high = data_frame.loc[row, 'Fresno']
            if data_frame.loc[row, 'San Francisco'] > sf_high :
                sf_high = data_frame.loc[row, 'San Francisco']
            if data_frame.loc[row, 'San Jose'] > sj_high :
                sj_high = data_frame.loc[row, 'San Jose']
            if data_frame.loc[row, 'Sacramento'] > sac_high :
                sac_high = data_frame.loc[row, 'Sacramento']
                
    X_by_week.loc[current_day] = [week_demand, pop, \
        la_high, sd_high, sj_high, sf_high, sac_high, f_high, current_day.weekday()]

                
    return X_by_week


def agg_daily(data_frame):
    '''
        Function: agg_daily

        Description: Aggregate the energy demand data on a daily basis.  
        Accumulate the energy demand into a daily total.
        Record the high temperature for each of the cities in the dataset.  
        Then create and return the new dataframe.

        Arguments: data_frame - Pandas dataframe containing the energy demand
            data on an hourly basis.

        Return: Aggregated dataframe with daily demand total and daily high temps.

    '''

    # Create the new dataframe with a single entry for the first day
    current_day = data_frame.index[0].date()
    X_by_day = pd.DataFrame([], index=[current_day], \
        columns=['day_demand', 'CA_pop', 'la_high', 'sd_high', 'sj_high', 'sf_high',\
                   'sac_high', 'f_high', 'DOW' ])

    # Initialize the variables used to accumulate daily data
    day_demand = 0.0 
    la_high = 0
    sd_high = 0
    f_high = 0
    sj_high = 0
    sf_high = 0
    sac_high = 0
    pop = 0


    # for each entry in the dataframe
    for row in data_frame.index :

        # Check the date for the next entry
        next_day = row.date()

        # If we're changing days then 
        if next_day != current_day :
            X_by_day.loc[current_day] = [day_demand, pop, la_high, sd_high, sj_high, sf_high, \
                        sac_high, f_high, current_day.weekday()]
            '''
            {'day_demand' :day_demand, 'la_high': la_high,\
                'sd_high' : sd_high, 'f_high' : f_high, 'DOW' : current_day.weekday()}
            '''
            current_day = next_day
            la_high = 0
            sd_high = 0
            sj_high = 0
            sf_high = 0
            sac_high = 0
            f_high = 0
            day_demand = 0
            pop=0

        # Accumulate the demand and test high temperatures.  If this is a new day the
        #  values were reset above.
        day_demand += data_frame.loc[row, 'demand']
        pop = data_frame.loc[row, 'CA_pop']
        if data_frame.loc[row, 'Los Angeles'] > la_high :
            la_high = data_frame.loc[row, 'Los Angeles']
        if data_frame.loc[row, 'San Diego'] > sd_high :
            sd_high = data_frame.loc[row, 'San Diego']
        if data_frame.loc[row, 'Fresno'] > f_high :
            f_high = data_frame.loc[row, 'Fresno']
        if data_frame.loc[row, 'San Jose'] > sj_high :
            sj_high = data_frame.loc[row, 'San Jose']
        if data_frame.loc[row, 'San Francisco'] > sf_high :
            sf_high = data_frame.loc[row, 'San Francisco']
        if data_frame.loc[row, 'Sacramento'] > sac_high :
            sac_high = data_frame.loc[row, 'Sacramento']
    
    # Save the last of the accumulated data
    X_by_day.loc[current_day] = [day_demand, pop, la_high, sd_high, sj_high, sf_high, \
                sac_high, f_high, current_day.weekday()]
    
    # Return the dataframe for the aggregated data.
    return X_by_day

    

def test_stationarity(timeseries, window = 12, cutoff = 0.01):
    '''
        Function: test_stationarity
            This fuction runs the Dickey-Fuller test for stationarity on the timeseries parameter.
        Parameters:
            timeseries - data for the dependent variable in the time series.
            window - size of the window used for statistical analysis
            cutoff - limit for the p-value from the Dickey-Fuller test to categorize data as stationary or not. 
        Return:
            None
    '''

    #Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC', maxlag = 20 )

    #Put the results in a Pandas series
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    # Print the individual results
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value

    # Test the p-value of 5% critical value
    pvalue = dftest[1]

    # If we're below the cutoff then the data is probably stationary
    if pvalue < cutoff:
        print('p-value = %.4f. The series is likely stationary.' % pvalue)
    else:
        print('p-value = %.4f. The series is likely non-stationary.' % pvalue)
    
    print(dfoutput)
    
    
    
def order_sweep(model, arima_order_list, seasonal_order_list, endo_data, exo_data=None) :
    '''
        Function: order_sweep
            This fuction conducts a series of SARIMAX model fits for different sets of
            order parameter configurations.  The SARIMAX model uses two different sets of
            parameters for model fit: arima order (p-d-q) and seasonal order (P-D-Q-S).
            The arima parameters are:
                p - number of trailing autoregressive steps to use
                d - number of differencing (discrete time differentiation) steps to use to make the data stationary.
                q - number of trailing error values to use in the moving average component of the model.
            The seasonl parameters are:
                P - number of trailing autoregressive seasonal components to include
                D - number of seasonal differencing steps to use
                Q - number of trailing seasonal errors to use in the moving average component
                s - number of time steps in the the 'season'
        
        Parameters:
            model - function that will be used for model definition and fit.
            arima_order_list - list of tuples containing the arima order parameters to test.
            seasonal_order_list - list of tuples containing the seasonal parameters to test.
            endo_data - endogenous time series data we are trying to model
            exo_data - exogenous data that supports the model development

        Return:
            aic_dict - Dictionary containing the parameters used for a model fit and the aic, bic and mse scores for the fit.

    '''
    aic_dict = defaultdict(dict)
    i=0
    for order in arima_order_list :
        for seasonal in seasonal_order_list :
            if exo_data is not None :
                mod = model(endo_data, order=order, seasonal_order=seasonal, exog=exo_data)
            else :
                mod = model(endo_data, order=order, seasonal_order=seasonal)
                
            try :
                res = mod.fit()
                aic_dict[i] = {'order': order, 'seasonal order': seasonal, 
                    'aic': res.aic, 'bic':res.bic, 'mse': res.mse}
            except :
                continue
                
            i += 1
    return aic_dict

def top_results(result_dict, test_key, count) :
    '''
        Function top_results
            Return the best results from the results dictionary based on the indicated score

        Parameters:
            result_dict - dictionary containing the results from the order_sweep test
            test_key - string containing the key used to evaluate the results.  Can be 'aic', 'bic', or 'mse'.
            count - number of results to return

        Return:
            sorted list of top results in order.  For each of these metrics lower values are better.

    '''

    # Create lists for storing the results
    top_result = [1000000]*count
    order_list = [0]*count
    ret_results = []

    # for each set of results
    for i in result_dict.keys() :
        # Check if the current score is lower than the largest in the current list
        if result_dict[i][test_key] < max(top_result) :
            # If we have an improvement then replace the largest value with the current results
            top_result[top_result.index(max(top_result))] = result_dict[i][test_key]
            # Save the index of the results
            order_list[top_result.index(max(top_result))] = i

    # Create a list of the top results from the saved index values        
    for idx in order_list :
        ret_results.append(result_dict[idx])

    # Sort and return the list of values based on the specified key    
    return sorted(ret_results, key=lambda x : x[test_key])


def prophet_walk_forward_val(data, start_size, val_window, regressors, steps) :
    '''
        Function: walk_forward_val
        
        Arguments:
            model - model class instantiator
            data - pandas dataframe containing data of interest
            start_size - initial number of rows for training
            val_window - size of the validation window for each trail
            regressors - list of regressors to add to the model
            steps - number of walk forward steps to run
        
        Return:
            mape_list - list of tupels with the last index for training window and mape for the window
    '''
    mape_list = []
    # for each steps
    for step in range(steps) :
        print('Step %d starting at index %d' % (step+1, start_size+step*val_window))
        # Get the block of data for training
        train_dat = data_block(data, 0, start_size+step*val_window)
        
        # Get a block of data for validation
        val_dat = data_block(data, start_size+step*val_window, val_window)
        
        # Instantiate the model
        m = Prophet(weekly_seasonality=True, yearly_seasonality=True)
        
        # Add regressors
        for reg in regressors :
            m.add_regressor(reg)
            
        # Fit the model
        m.fit(train_dat)
        
        # Forecast for the validation step
        forecast = m.predict(val_dat)
        forecast.index = val_dat.index
        print(forecast['ds'].values[0], val_dat['ds'].values[0])
        
        # Caluclate the MAPE for the window
        ape = [np.abs(val_dat.loc[x, 'y'] - forecast.loc[x, 'yhat']) / val_dat.loc[x, 'y'] * 100 for x in val_dat.index]
        
        # Add the values to the list
        mape_list.append((train_dat.index[-1], np.mean(ape)))
        
    # return the mape list
    return mape_list

def sarimax_walk_forward_val(data, start_size, val_window, regressors, steps) :
    '''
        Function: sarimax_walk_forward_val
        
        Arguments:
            model - model class instantiator
            data - pandas dataframe containing data of interest
            start_size - initial number of rows for training
            val_window - size of the validation window for each trail
            regressors - list of regressors to add to the model
            steps - number of walk forward steps to run
        
        Return:
            mape_list - list of tupels with the last index for training window and mape for the window
    '''
    # Now choose the results that seem to work the best above and fit the model

    best_order3 = (5, 0, 0)
    best_seasonal_order3 = (3, 0, 1, 7)
    
    mape_list = []
    # for each steps
    for step in range(steps) :
        print('Step %d starting at index %d' % (step+1, start_size+step*val_window))
        # Get the block of data for training
        train_dat = data_block(data, 0, start_size+step*val_window)
        
        # Get a block of data for validation
        val_dat = data_block(data, start_size+step*val_window, val_window)
        

        model = SARIMAX(data['log_demand'], order=best_order3, \
                    seasonal_order=best_seasonal_order3, exog=data[regressors])
        res = model.fit()
        
        # Forecast for the validation step
        predict = res.get_forecast(val_window, exog=val_dat[regressors].values[:val_window, :])
        
        forecast = np.exp(predict.predicted_mean)
        
        # Caluclate the MAPE for the window
        ape = [np.abs(val_dat['day_demand'].values[x] - forecast[x]) / \
               val_dat['day_demand'].values[x] * 100 for x in range(len(val_dat.index))]
        
        # Add the values to the list
        mape_list.append((train_dat.index[-1], np.mean(ape)))
        
    # return the mape list
    return mape_list

def data_block(X, start, out_size) :
    # get the slices of data frames to provide the blocks requested
    block1 = X.iloc[start:start+out_size, :]
        
    return block1
 
 
