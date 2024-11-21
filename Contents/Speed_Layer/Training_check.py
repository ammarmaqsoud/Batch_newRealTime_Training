import math

import numpy as np
import pandas as pd
import seaborn as sns
# from keras.losses import mean_squared_error
from sklearn.metrics import mean_squared_error,r2_score

sns.set_style('whitegrid')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense,LSTM, GRU, Dropout
from sklearn.preprocessing import MinMaxScaler
print("loaded")

from keras.models import load_model
# model = load_model('8_11_train_test_40_20_epoch_25_batch_8_native.h5')
model = load_model('Models/onlineModel_4.h5')
# model = load_model('22_11_train_test_20_20_epoch_25.h5')
model.compile(optimizer= 'adam', loss= 'mse' )


data_dir='../Data/DailyDelhiClimateTrain.csv'
df=pd.read_csv(data_dir)

df['date']=pd.to_datetime(df['date'])
df.set_index('date',inplace=True)

n_cols = 1

#=============================RMSE============================
df = pd.read_csv("../Data/DailyDelhiClimateTrain.csv",
                 parse_dates=['date'],  # change to date time format
                 index_col="date")

df = df[['meantemp']]

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
dl_train, dl_test = df.iloc[:train_size], df.iloc[train_size:]
print(len(dl_train), len(dl_test))
scaler = MinMaxScaler(feature_range=(0,1))


from sklearn.preprocessing import RobustScaler, MinMaxScaler

robust_scaler = RobustScaler()   # scaler for wind_speed
minmax_scaler = MinMaxScaler()  # scaler for humidity
target_transformer = MinMaxScaler()   # scaler for target (meantemp)

dl_train['meantemp_diff'] = dl_train['meantemp'].diff().fillna(0)
dl_test['meantemp_diff'] = dl_test['meantemp'].diff().fillna(0)


dl_train['meantemp'] = target_transformer.fit_transform(dl_train[['meantemp']]) # target
dl_test['meantemp'] = target_transformer.fit_transform(dl_test[['meantemp']])

dl_train['meantemp_diff'] = target_transformer.fit_transform(dl_train[['meantemp_diff']]) # target
dl_test['meantemp_diff'] = target_transformer.fit_transform(dl_test[['meantemp_diff']])

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Create sequences
sequence_length = 30  # Example sequence length (adjust based on your data and experimentation)
X_train, y_train = create_dataset(dl_train, dl_train['meantemp'], sequence_length)
X_test, y_test = create_dataset(dl_test, dl_test['meantemp'], sequence_length)


# Reshaping the input to (n_samples, time_steps, n_feature)
x_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))

# Creating a testing set with 60 time-steps and 1 output
time_steps = 30

# Get Prediction
predictions = model.predict(x_test)


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
#=============================RMSE============================

# # Start with the first value of the original series
# reconstructed_ts = [dl_test.iloc[0]]  # Starting point, the first value of the original series
#
# # Perform cumulative summation to reconstruct the original series
# for diff in y_test[1:]:  # Skip the first NaN value
#     reconstructed_ts.append(reconstructed_ts[-1] + diff)


#===========================PLOT=============================
plt.plot(predictions[:,0], color="blue",
         label="Predicted meantemp ", linewidth=2)
plt.plot(y_test[:,0], color="red",
         label="Actual meantemp ", linewidth=2)
plt.legend()
# Show the major grid lines with dark grey lines
plt.grid(visible=True, which="major", color="#666666", linestyle="-")
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(visible=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
plt.show()
#===========================PLOT=============================






