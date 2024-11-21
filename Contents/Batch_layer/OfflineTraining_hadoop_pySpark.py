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

import numpy as np
import pandas as pd
import seaborn as sns

import csv
import math
import os
import sys
import tensorflow as tf
from pandas import DataFrame
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import csv


os.environ['SPARK_HOME']="/opt/spark"
import findspark
findspark.init()

sys.path.append("/opt/spark/python")
sys.path.append("/opt/spark/python/lib/py4j-0.10.7-src.zip")

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    print ("Successfully imported Spark Modules")
except ImportError as e:
    print ("Can not import Spark Modules", e)



# from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.sql.functions import lag, col
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import lit
from pyspark.sql.window import Window
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from datetime import datetime
#from var import fitVar
import pyspark.sql.functions
import pyspark.sql.functions as F
import pyspark.sql.types as tp
import numpy as np
import sys
import os



os.environ['SPARK_HOME']="/opt/spark"
import findspark
findspark.init()

sys.path.append("/opt/spark/python")
sys.path.append("/opt/spark/python/lib/py4j-0.10.7-src.zip")

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    print ("Successfully imported Spark Modules")
except ImportError as e:
    print ("Can not import Spark Modules", e)




from pyspark.sql import SparkSession
from pyspark.sql.functions import lag, col
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import lit
from pyspark.sql.window import Window
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from datetime import datetime
import pyspark.sql.functions
import pyspark.sql.functions as F
import pyspark.sql.types as tp
import numpy as np
import sys
import os

sns.set_style('whitegrid')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional, Conv2D, Conv1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler, MinMaxScaler
print("loading...")

os.environ['PYSPARK_SUBMIT_ARGS'] = "--packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.3 pyspark-shell"

sparkSession = SparkSession.builder.appName("example-pyspark-read-and-write").getOrCreate()
# Read stream from HDFS.
df_load = sparkSession.read.csv('hdfs://127.0.0.1:9000/Training_Stream/')
df_load.show(10)
print((df_load.count(), len(df_load.columns)))

temperatures = []
_date = []

floatTemperatures = np.array(temperatures, dtype=np.float32)
str_dates = np.array(_date, dtype=np.str)

# Using pandas
import pandas as pd

sparkSession.conf.set("spark.sql.execution.arrow.enabled", "true")
i = 0
pandasDF = df_load.toPandas()


str_dates = np.append(str_dates, "2013-01-01")  # add last temperature
floatTemperatures = np.append(floatTemperatures, 10.0)  # add last temperature


for index, row in pandasDF.iterrows():
    if i == 0:
        i = i + 1
        continue
    str_dates = np.append(str_dates, row['_c0'].split(',')[0])
    floatTemperatures = np.append(floatTemperatures, row['_c0'].split(',')[1])

floatTemperatures = np.append(floatTemperatures, 10.0)  # add last temperature
str_dates = np.append(str_dates, "2017-01-01")  # add last temperature


floatTemperatures = floatTemperatures.astype("float32")

my_dataset_1 = pd.DataFrame(str_dates)  # Convert numpy array into pandas dataframe
my_dataset_2 = pd.DataFrame(floatTemperatures)  # Convert numpy array into pandas dataframe

my_dataset_2 = my_dataset_2.astype("float32")


df = pd.concat([my_dataset_1, my_dataset_2], axis=1)
df.columns = ['date', 'meantemp']



# Split the data to training and testing
train_size = int(len(df) * 0.8)
dl_train, dl_test = df.iloc[:train_size], df.iloc[train_size:]
print(len(dl_train), len(dl_test))


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
X_train, y_train = create_dataset(dl_train[['meantemp','meantemp_diff']], dl_train['meantemp'], sequence_length)
X_test, y_test = create_dataset(dl_test[['meantemp','meantemp_diff']],dl_test['meantemp'], sequence_length)

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
model.save("Models/17_11_offlineTraining_hadoop_pySpark_epoch_20.h5")

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Validation Loss: {loss}')


# from keras.losses import mean_squared_error
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
# Make predictions
predictions = model.predict(X_test)
predictions = target_transformer.inverse_transform(predictions)  # Inverse transform to original scale

# Inverse transform the true values for comparison
y_test = y_test.reshape(-1, 1)
y_test = target_transformer.inverse_transform(y_test)


#  RMSE and R2 scores
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