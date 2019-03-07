import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from alpha_vantage.timeseries import TimeSeries
import keras.backend as K


## Loading prices
 
def loading_mid_prices(alpha_vantage_key, to_use, to_predict):
    """
    Loading the data in a dataframe containing the mid-daily price for each considered stock:
    
    - alpha_vantage_key: the key to have access to the API
    - to_use: the list of wanted stock symbols
    """
    
    ts = TimeSeries(key=alpha_vantage_key, output_format='pandas')
    df_list = []
    for stock in to_use:
        print('Processing ' + stock)
        df, meta_data = ts.get_daily(symbol=stock, outputsize='full')
        df.columns = ['Date','Low','High','Close','Open']
        df[stock] = (df['High'] +df['Low'])/ 2.0
        df.set_index('Date')
        df = df.drop(['Low','High','Close','Open', 'Date'], axis=1)
        df_list.append(df)
    final_df = pd.concat(df_list, axis=1, sort=True)
    index_to_cut = look_for_first_nan_in_column(to_predict, final_df)
    return final_df[index_to_cut:]


def look_for_first_nan_in_column(stock_to_predict, df):
    for i in range(len(df)):
        if not np.isnan(df[stock_to_predict].iloc[i]):
            return i


## Data vizualization

def visualize_stocks(df):
    """
    Vizualize the different stock prices in a graph
    - df: previously built dataframe (using the loading mid-price function)
    """
    
    plt.figure(figsize = (10,5))
    for stock in list(df.columns):
        index = list(df.index)
        plt.plot(list(range(len(index))), df[stock], label=stock)
        plt.xticks(range(0, len(index),100), np.take(index, list(range(0, len(index), 100))), rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Average Daily Price')
        plt.legend(loc='best')


## Building the dataset

def from_serie_to_xy(time_serie, x_len, y_len, return_y = False):
    """ 
    Create from a time serie a list of regression samples composed of x_len numbers in and y_len numbers out
    - time_serie: original time serie we would like to transform into a regression set
    - x_len: number of prices we want to do the prediction
    - y_len: number of prices we want to predict
    - return_y: set True if we want to get the regression targets, False otherwise 
    (if this time serie is divided in such a set only for predicting prices of another time serie)
    """
    
    n = len(time_serie)        
    x = []
    y = []
    for i in range(0,n-(x_len+y_len)+1):
            x.append(np.array(time_serie[i:i+x_len]))
            y.append(np.array(time_serie[i+x_len:i+x_len+y_len]))
    if return_y:
        return (x, y)
    else:
        return x



def adjust_series_to_same_length(stock_ref, prices_df):
    """Time series of stock do not always have the same length as they were not introduced in the stock market at the same time.
    This function adjust them at the same length by taking the average of the ten first prices if it is too short, 
    or by cutting the time serie if it is too long.
    - stock_ref: time serie of the stock whose price we want to predict
    - prices_df: dataframe of all the time series of stocks we want to adjust
    """
    
    df = prices_df.copy()
    total_len = df.shape[0]
    indexes = list(df.index)
    idx_ref = df[stock_ref].first_valid_index()
    usable_ref = indexes.index(idx_ref)
    ref_len = total_len - usable_ref
    df = df.iloc[total_len-ref_len:, :]
    for stock in df.columns:
        idx = df[stock].first_valid_index()
        usable = indexes.index(idx)
        mean = np.mean(np.array(df[stock].iloc[usable:usable+10]))
        df[stock] = df[stock].fillna(mean)
    return df


    
def states_up_down(serie, last_number):
    """ A tool function """
    if serie[-1] > last_number:
        return 0
    else:
        return 1



def create_dataset(price_df, stock_to_predict, model='exact_numbers', x_len=30, y_len=1):
    """
    Create a regression/classification dataset
    - price_df: dataframe containing the different stock prices we are goig to use to make our predictions
    - stock_to_predict: stock whose prices we are going to predict
    - model: 'exact_number' (create a regression dataset with y_len prices to predict)
             'up_or_down' (create a classification dataset with a 1 if the next price is going to go up, a 0 otherwise)
    - x_len: number of prices we want to do the prediction
    - y_len: number of prices we want to predict or moment we want to predict when the price will go up or down
    """
    
    
    df = price_df.copy()
    #let first adjust all time series to the same size
    df = adjust_series_to_same_length(stock_to_predict, df)
    
    #then create dataframe
    if model=='exact_numbers':
        dff = pd.DataFrame()
        for stock in df.columns:
            if stock == stock_to_predict:
                listes = from_serie_to_xy(list(df[stock]), x_len, y_len, True)
                dff = pd.concat([dff, pd.Series(listes[0], name=stock), pd.Series(listes[1], name="y_"+stock)], axis = 1)
            else:
                listes = from_serie_to_xy(list(df[stock]), x_len, y_len)
                dff = pd.concat([dff, pd.Series(listes, name=stock)], axis = 1)
        return dff
    
    elif model=='up_or_down':
        dff = pd.DataFrame()
        for key in df.columns:
            if key == stock_to_predict:
                listes = from_serie_to_xy(list(df[key]), x_len, y_len, True)
                zipped = list(zip(listes[0], listes[1]))
                y  = []
                for couple in zipped:
                    y.append(states_up_down(couple[0], couple[1][-1]))
                dff = pd.concat([dff, pd.Series(listes[0], name=key), pd.Series(y, name="y_"+key)], axis = 1)
            else:
                listes = from_serie_to_xy(list(df[key]), x_len, y_len)
                dff = pd.concat([dff, pd.Series(listes, name=key)], axis = 1)
        return dff
    
    else:
        print("Model can only be 'up_or_down' or 'exact_numbers'")


## Splitting in train test val

def split_train_val_test(df, val_size=0.15, test_size=0.15, model='shuffle', nb_period=10, seed=0):
    """
    Creating the training, validation and test sets
    - df: dataframe as created by the create_dataset function
    - val_size: size of the validation set
    - test_size: size of the test set
    - model : 'shuffle': all samples are shuffled, we are training the model on data from every period
              'time-consistent': we are training on a period, validating of the future of training period and 
                                 testing on the future of this validation period
              'alternate': training, validation and test are "rotative" ranges. For instance,
                           we train on 2004, 2007, 2010, validate on 2005, 2008, 2011 and test on 2006, 2009, 2012
    - nb_period: only if model=='alternate'. Number of rotations to do. (3 in the previous example)
    - seed: only if model=='shuffle'. A random seed in order to be able to get exactly the same dataset if needed again.
    """
    
    if model == 'shuffle':
        l1 = train_test_split(df, test_size=val_size+test_size, shuffle=True, random_state=seed)
        l2 = train_test_split(l1[1], test_size = test_size/(val_size+test_size), shuffle=True, random_state=seed)
        return (l1[0], l2[0], l2[1])
    elif model == "time-consistent":
        l1 = train_test_split(df, test_size=val_size+test_size, shuffle=False)
        l2 = train_test_split(l1[1], test_size = test_size/(val_size+test_size), shuffle=False)
        return (l1[0], l2[0], l2[1])
    elif model == 'alternate':
        size = int(df.shape[0]/5)
        train = pd.DataFrame()
        val  = pd.DataFrame()
        test = pd.DataFrame()
        for i in range(nb_period):
            try: 
                sample = df.iloc[i*size:(i+1)*size,:]
            except:
                sample = df.iloc[i*size:,:]
            l1 = train_test_split(sample, test_size=val_size+test_size, shuffle=False)
            l2 = train_test_split(l1[1], test_size = test_size/(val_size+test_size), shuffle=False)
            train = pd.concat([train, l1[0]], axis=0)
            val = pd.concat([val, l2[0]], axis=0)
            test = pd.concat([test, l2[1]], axis=0)
        return (train, val, test)
    else:
        print("Model can only be 'shuffle', 'alternate' or 'time-consistent'")


## Vizualize predictions

def visualize_predictions_training(df, df_pred, x_len, y_len, mode_pred='regression'):
    """
    To test this function below, I replace 'predictions' with 'y_MSFT' as obtained in the above df
    This function is working for the 'time-consistent' and 'alternate' models but not always for the 'shuffle' one.
    - df: original dataframe as obtained with the loading_mid_prices function
    - df_pred: dataframe containing as index the same one than the set on which pred were made and as value the predictions
    - x_len: number of prices we want to do the prediction
    - y_len: number of prices we want to predict
    - mode_pred: 'regression' or 'classificaton' according to what was done
    
    This function shows one pred out of y_len
    """
    
    plt.figure(figsize = (10,5))
    for stock in list(df.columns):
        index = list(df.index)
        plt.plot(list(range(len(index))), df[stock], label=stock)
        plt.xticks(range(0, len(index),100), np.take(index, list(range(0, len(index), 100))), rotation=45)

    
    if mode_pred == 'regression':
        abscisses = np.array(df_pred.index)
        ordonnees = []
        pred = list(df_pred['predictions'])
        for i in range(0,len(pred),y_len):
            for j in range(y_len):
                ordonnees.append(pred[i][j])
        ordonnees = ordonnees[0:len(abscisses)]
        plt.scatter(abscisses, np.array(ordonnees), label='predictions', s=1, c='b')
        
    elif mode_pred == 'classification':
        abscisses = np.array(df_pred.index)
        ordonnees = []
        pred = list(df_pred['predictions'])
        for i in range(len(pred)):
            if pred[i] == 1:
                try:
                    ordonnees.append(ordonnees[i-1]+5)
                except:
                    ordonnees.append(4)
            elif pred[i] == 0:
                try:
                    ordonnees.append(ordonnees[i-1]-5)
                except:
                    ordonnees.append(-4)
        ordonnees = ordonnees[0:len(abscisses)]
        plt.scatter(abscisses, np.array(ordonnees), label='predictions', s=1, c='b')
    
    plt.xlabel('Day number')
    plt.ylabel('Average Daily Price')
    plt.legend(loc='best')

## Data for new prediction

def data_for_prediction(to_predict, dataframe, x_len):
    total = []
    for i in list(dataframe.index)[-x_len:]:
        data = list(dataframe.loc[i])
        interm = []
        for j in range(len(data)):
            interm.append(data[j])
        interm = np.array(interm)
        total.append(interm)
    return np.array([np.array(total)])



## Visualize prediction

def add_predictions_to_vizualization(prediction, dataframe, to_predict):
    df = pd.DataFrame({to_predict:list(prediction[0])})
    df_concat = pd.concat([dataframe, df], axis=0, sort=False)
    visualize_stocks(df_concat)
    return df_concat
    

## Reshape for network

def to_multidim_array_train(dataframe):
    total = []
    for i in list(dataframe.index):
        data = list(dataframe.loc[i])
        interm = []
        for j in range(len(data[0])):
            interm.append(np.array(data)[:,j])
        interm = np.array(interm)
        total.append(interm)
    return np.array(total)

def to_multidim_array_y(dataframe):
    data = []
    for i in list(dataframe.index):
        data.append(np.array(list(dataframe.loc[i])))
    return np.array(data)

def to_onedim_array(liste):
    new = []
    for i in range(len(liste)):
        new.append(list(liste[i]))
    return new

## Metrics


def nmse_metric(y_true, y_pred):
    mse = K.mean((y_pred - y_true)**2)
    true_norm = K.mean((y_true)**2)
    return mse/true_norm

def std_diff_metric(y_true, y_pred):
    return K.std(y_pred)

def nmse_metric_for_np(y_true, y_pred):
    mse = (np.linalg.norm(y_pred - y_true))**2
    true_norm = (np.linalg.norm(y_true))**2
    return mse/true_norm


