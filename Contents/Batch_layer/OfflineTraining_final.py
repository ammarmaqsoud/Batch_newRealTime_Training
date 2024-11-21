import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

import plotly
import plotly.express as px
import plotly.graph_objects as go
import cufflinks as cf


import warnings

from keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM

warnings.filterwarnings("ignore")
warnings.warn("this will not show")

plt.rcParams["figure.figsize"] = (10,6)

sns.set_style("whitegrid")
pd.set_option('display.float_format', lambda x: '%.3f' % x)





pd.set_option('display.max_columns', None)


df = pd.read_csv("../Data/DailyDelhiClimateTrain.csv",
                 parse_dates=['date'],  # change to date time format
                 index_col="date")

df = df[['meantemp']]


# Split the data to training and testing
train_size = int(len(df) * 0.8)
dl_train, dl_test = df.iloc[:train_size], df.iloc[train_size:]
print(len(dl_train), len(dl_test))



from sklearn.preprocessing import RobustScaler, MinMaxScaler



target_transformer = MinMaxScaler()

dl_train['meantemp_diff'] = dl_train['meantemp'].diff().fillna(0)
dl_test['meantemp_diff'] = dl_test['meantemp'].diff().fillna(0)


dl_train['meantemp'] = target_transformer.fit_transform(dl_train[['meantemp']]) # target
dl_test['meantemp'] = target_transformer.fit_transform(dl_test[['meantemp']])

dl_train['meantemp_diff'] = target_transformer.fit_transform(dl_train[['meantemp_diff']]) # target
dl_test['meantemp_diff'] = target_transformer.fit_transform(dl_test[['meantemp_diff']])


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



sequence_length = 30
X_train, y_train = create_dataset(dl_train, dl_train['meantemp'], sequence_length)
X_test, y_test = create_dataset(dl_test, dl_test['meantemp'], sequence_length)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))
X_test= np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))


from tensorflow.keras.layers import GRU

model = Sequential()
model.add(GRU(100, activation='tanh', input_shape=(sequence_length, X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')



# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=8, callbacks=[early_stopping])
model.save("Models/17_11_offlineTraining_epoch_20.h5")

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Validation Loss: {loss}')



from sklearn.metrics import  mean_squared_error,r2_score, mean_absolute_error
# Make predictions
predictions = model.predict(X_test)
predictions = target_transformer.inverse_transform(predictions)  # Inverse transform to original scale


y_test = y_test.reshape(-1, 1)
y_test = target_transformer.inverse_transform(y_test)


#  RMSE  R2 scores
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f'RMSE: {rmse}')
print(f'R2 Score: {r2}')



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