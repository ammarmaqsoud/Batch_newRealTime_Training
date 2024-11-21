import numpy as np
import pandas as pd
import seaborn as sns
from keras.losses import mean_squared_error
from sklearn.metrics import r2_score

sns.set_style('whitegrid')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional, Conv2D, Conv1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
print("loaded")

data_dir='../Data/DailyDelhiClimateTrain.csv'
df=pd.read_csv(data_dir)

df['date']=pd.to_datetime(df['date'])
df.set_index('date',inplace=True)


df = df[['meantemp']]



train_size = int(len(df) * 0.20)
test_size=len(df)-int(len(df)*0.80)
# dl_train, dl_test = df.iloc[:train_size], df.iloc[train_size:]

dl_train, dl_test = df.iloc[0:train_size,:], df.iloc[int(len(df)*0.80):]
print(len(dl_train), len(dl_test))





from sklearn.preprocessing import RobustScaler, MinMaxScaler

robust_scaler = RobustScaler()   # scaler for wind_speed
minmax_scaler = MinMaxScaler()  # scaler for humidity
target_transformer = MinMaxScaler()   # scaler for target (meantemp)

dl_train['meantemp_diff'] = dl_train['meantemp'].diff().fillna(0)
dl_test['meantemp_diff'] = dl_test['meantemp'].diff().fillna(0)


dl_train['meantemp'] = target_transformer.fit_transform(dl_train[['meantemp']]) # target
dl_test['meantemp'] = target_transformer.transform(dl_test[['meantemp']])

dl_train['meantemp_diff'] = target_transformer.fit_transform(dl_train[['meantemp_diff']]) # target
dl_test['meantemp_diff'] = target_transformer.transform(dl_test[['meantemp_diff']])


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)



sequence_length =30
X_train, y_train = create_dataset(dl_train, dl_train['meantemp'], sequence_length)
X_test, y_test = create_dataset(dl_test, dl_test['meantemp'], sequence_length)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))
X_test= np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))











#
# # Build the bidirectional LSTM model
model = Sequential()
model.add(GRU(100, activation='tanh', input_shape=(sequence_length, X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(30, 2)))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Bidirectional(LSTM(100, activation='tanh')))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')

model.summary()


model.compile(optimizer= 'adam', loss= 'mse' )

# model.compile(loss="mean_squared_error", optimizer='adam', metrics=["mse","mae"])

# history = model.fit(X_train,y_train, epochs= 25 ,validation_data=(X_test, y_test) ,batch_size=1)
history = model.fit(X_train,y_train, epochs= 25  ,batch_size=1)
model.save("22_11_train_test_20_20_epoch_25.h5")


plt.figure(figsize=(12, 8))
plt.plot(history.history["loss"])
plt.legend(['Mean Squared Error','Mean Absolute Error'])
plt.title("Losses")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

# Get training and validation losses from history
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Plot loss values over epochs
plt.figure(figsize=(12, 8))
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()



# Get Prediction
predictions = model.predict(X_test)


# inverse prediction scalling
predictions=target_transformer.inverse_transform(predictions)
print(predictions.shape)

#inverse y_test scaling
y_test = y_test.reshape(-1, 1)
y_test = target_transformer.inverse_transform(y_test)

# Calculate RMSE and R2 scores
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f'RMSE: ', rmse.mean())
print(f'R2 Score: {r2}')




