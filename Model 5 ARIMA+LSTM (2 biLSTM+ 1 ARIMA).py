from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import adfuller

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout,Bidirectional,BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import layers


def transform(df):
    df = df.sort_index()
    df['Change %'] = df['Price'].pct_change()
    df = df.rename(columns={"Change %": "pct_change"})
    df['Log_Return'] = np.log(df['Price']/df['Price'].shift(1))
    vol=[]
    for i in df['Vol.']:
        if i[-1] == 'M':
            i = float(i[:-1])*1000000
        elif i[-1] == 'K':
            i = float(i[:-1])*1000
        else :
            i = 0   
        vol.append(i)
    df['Vol.'] = vol
    return df

def split(df):
    train_len=1000
    test_len=125
    data = df.tail(test_len + train_len)
    train, test = data.head(train_len), data.tail(test_len)
    return train, test

def get_arima(data, train_len, test_len):
    # prepare train and test data
    data = data.tail(test_len + train_len).reset_index(drop=True)
    train = data.head(train_len).values.tolist()
    test = data.tail(test_len).values.tolist()

    # Initialize model
    model = auto_arima(train, max_p=3, max_q=3, d=1, seasonal=False, trace=True,
                       error_action='ignore', suppress_warnings=True)

    # Determine model parameters
    model_fit = model.fit(train)
    order = model.get_params()['order']
    print('ARIMA order:', order, '\n')
        
    # Genereate predictions
    predictions = []
    for i in range(len(test)):
        model = ARIMA(train,order=order)
        model_fit = model.fit()
        print('working on', i+1, 'of', test_len, '-- ' + 
              str(int(100 * (i + 1) / test_len)) + '% complete')
        predictions.append(model_fit.predict()[0])     
        train.append(test[i])
                  
    # Residuals data
    mod = ARIMA(data,order=order)
    results = mod.fit()
    residuals = results.resid
    s1=pd.Series(0)
    residuals = s1.append(residuals)
    return predictions, residuals


def plot_arima(train_data,test_data,predictions):
    plt.figure(figsize=(16,8))
    plt.plot(train_data.index, train_data.Price, color='green', label = 'Train Crypto Price')
    plt.plot(test_data.index, test_data.Price, color = 'red', label = 'Real Crypto Price')
    plt.plot(test_data.index, predictions, color = 'blue', label = 'Predicted Crypto Price')
    plt.legend()
    plt.grid(True)
    plt.show()
            
    mse = mean_squared_error(test_data.Price, predictions)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(test_data.Price, predictions)  
    
    print('MSE: ',mse)
    print('RMSE: ',rmse)
    print('MAPE: ',mape)
    
    
def price_feature(train_data,test_data):
    train = train_data.iloc[:, 0:1].values  
    test = test_data.iloc[:, 0:1].values 
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    
    timesteps = 7
    X_train = []
    y_train = []
    for i in range(timesteps, train.shape[0]):
        X_train.append(train_scaled[i-timesteps:i, 0]) #
        y_train.append(train_scaled[i, 0]) 
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))#
    combine = pd.concat((train_data['Price'], test_data['Price']), axis = 0) #
    test_inputs = combine[len(combine) - len(test_data) - timesteps:].values
    test_inputs = test_inputs.reshape(-1,1) 
    test_inputs = scaler.transform(test_inputs) 
    y_scaler = scaler.fit(train[:,0].reshape(-1,1))
    
    X_test=[]
    y_test=[]
    for i in range(timesteps, test_data.shape[0]+timesteps): 
        X_test.append(test_inputs[i-timesteps:i, 0]) #
        y_test.append(test_inputs[i, 0])
    X_test ,y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) #
    
    return X_train, y_train, X_test, y_test, y_scaler


def plot_results(train_data,test_data,predicted):    
    plt.figure(figsize=(14,8))
    plt.plot(train_data.index, train_data.Price, color='green', label = 'Train Crypto Price')
    plt.plot(test_data.index, test_data.Price, color = 'red', label = 'Real Crypto Price')
    plt.plot(test_data.index, predicted, color = 'blue', label = 'Predicted Crypto Price')
    plt.legend()
    plt.grid(True)
    plt.show()           
    mse = mean_squared_error(test_data.Price, predicted)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(test_data.Price, predicted)      
    print('MSE: ',mse)
    print('RMSE: ',rmse)
    print('MAPE: ',mape)

def plot_results_diff(train_data,test_data,predicted):    
    plt.figure(figsize=(14,8))
    plt.plot(train_data.index, train_data['diff'], color='green', label = 'Train Crypto Price')
    plt.plot(test_data.index, test_data['diff'], color = 'red', label = 'Real Crypto Price')
    plt.plot(test_data.index, predicted, color = 'blue', label = 'Predicted Crypto Price')
    plt.legend()
    plt.grid(True)
    plt.show()           
    mse = mean_squared_error(test_data['diff'], predicted)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(test_data['diff'], predicted)      
    print('MSE: ',mse)
    print('RMSE: ',rmse)
    print('MAPE: ',mape)    
    
    
def differencing(df):
    price_log = np.log(df.Price)
    price_log_diff = np.diff(price_log)
    diff = np.insert(price_log_diff, 0, 0)
    df['diff'] = diff

def check_stationarity(timeseries):
    result = adfuller(timeseries,autolag='AIC')
    dfoutput = pd.Series(result[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    print('The test statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('%s: %.3f' % (key, value))

def inv_diff(diff_df, first_value, add_first=True):
    """
    The index of the difference sequence starts from 1
    """
    diff_df.reset_index(drop=True, inplace=True)
    #print(diff_df)
    diff_df.index = diff_df.index + 1
    #print(diff_df)
    diff_df = pd.DataFrame(diff_df)
    diff_df = diff_df.cumsum()
    df = diff_df + first_value
    if add_first:
        df.loc[0] = first_value
        df.sort_index(inplace=True)
    return df


def create_model5(X_train, X_train2, X_test, X_test2, y_train, y_test, re_train, re_test, y_scaler):
    A_input=layers.Input(shape=(X_train.shape[1], X_train.shape[2],))
    A_B = layers.Bidirectional(LSTM(128,activation='relu',recurrent_dropout=0.2))(A_input)
    A_D = layers.Dense(64,activation='relu')(A_B)
    A_D2 = layers.Dense(10,activation='relu')(A_D)
    
    B_input= layers.Input(shape=(X_train2.shape[1], X_train2.shape[2],))
    B_B = layers.Bidirectional(LSTM(128,activation='relu',recurrent_dropout=0.2))(B_input)
    B_D = layers.Dense(64,activation='relu')(B_B)
    B_D2 = layers.Dense(10,activation='relu')(B_D)
    
    C_input = layers.Input(shape=(1,))
    C_bn = layers.BatchNormalization()(C_input)
    
    concate = layers.Concatenate()([A_D2, B_D2, C_bn])
    dense = layers.Dense(32, activation='relu')(concate)
    pred = layers.Dense(1, name='output')(dense)
    
    model5=Model(inputs=[A_input,B_input, C_input], outputs = pred)
    model5.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, mode='min')
    mc5 = None
    mc5 = ModelCheckpoint('best_model5.h5',save_best_only = True,verbose = 1,mode = 'min', monitor='val_loss')
    
    model5.fit([X_train, X_train2, re_train[7:]], y_train, epochs=10, 
          batch_size=32,callbacks=[reduce_lr,mc5], validation_data=([X_test, X_test2, re_test], y_test))
    
    best = load_model('best_model5.h5')
    predicted= best.predict([X_test, X_test2, re_test])
    predicted= y_scaler.inverse_transform(predicted)   
    return predicted 




