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
        print(start_size+step*val_window)
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

def data_block(X, start, out_size) :
    # get the slices of data frames to provide the blocks requested
    block1 = X.iloc[start:start+out_size, :]
        
    return block1
 
 
