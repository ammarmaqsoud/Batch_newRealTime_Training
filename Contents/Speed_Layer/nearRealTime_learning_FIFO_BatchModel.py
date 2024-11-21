import math
import os
import sys
from subprocess import Popen, PIPE

import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, Bidirectional, LSTM, MaxPooling1D, Dense, GRU
# from keras.losses import mean_squared_error
from pandas import DataFrame
from sklearn.metrics import mean_squared_error,r2_score
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
from sklearn.preprocessing import RobustScaler,MinMaxScaler


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



target_transformer = MinMaxScaler()

dl_train['meantemp_diff'] = dl_train['meantemp'].diff().fillna(0)
dl_test['meantemp_diff'] = dl_test['meantemp'].diff().fillna(0)


dl_train['meantemp'] = target_transformer.fit_transform(dl_train[['meantemp']]) # target
dl_test['meantemp'] = target_transformer.fit_transform(dl_test[['meantemp']])

dl_train['meantemp_diff'] = target_transformer.fit_transform(dl_train[['meantemp_diff']]) # target
dl_test['meantemp_diff'] = target_transformer.fit_transform(dl_test[['meantemp_diff']])

def Generate_dataset(x, y, time_steps=1):
    xs, ys = [], []
    for i in range(len(x) - time_steps):
        temp = x.iloc[i:(i + time_steps)].values
        xs.append(temp)
        ys.append(y.iloc[i + time_steps])
    return np.array(xs), np.array(ys)


sequence_length = 30
X_train, y_train = Generate_dataset(dl_train, dl_train['meantemp'], sequence_length)
X_test, y_test = Generate_dataset(dl_test, dl_test['meantemp'], sequence_length)


x_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))


time_steps = 30


y_test = y_test.reshape(-1, 1)
y_test = target_transformer.inverse_transform(y_test)

#=============================RMSE============================
def near_RealTime_RMSE():
    global x_test
    predictions = model.predict(x_test)


    predictions = target_transformer.inverse_transform(predictions)
    print(predictions.shape)

    # RMSE and R2 scores
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f'RMSE: {rmse}')
    print(f'R2 Score: {r2}')

#================================================================================

os.environ['PYSPARK_SUBMIT_ARGS'] = "--packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.3 pyspark-shell"
from pyspark.sql import SparkSession



from tensorflow.keras.models import Sequential



################################ FIFO 30 items window for link th Batches (Stateful)##################################
class FIFOQueue:
    def __init__(self, max_length):
        self.max_length = max_length
        self.queue = pd.DataFrame(columns=['data'])
    def enqueue(self, item):
        # Append the new item to the DataFrame
        self.queue = self.queue.append({'data': item}, ignore_index=True)

        # Check if the length exceeds the max_length
        if len(self.queue) > self.max_length:
            self.queue = self.queue.iloc[1:].reset_index(drop=True)  # Remove the oldest item

    def __repr__(self):
        return repr(self.queue)


    def is_empty(self):
        return len(self.queue) == 0
out=0

fifo_window = FIFOQueue(max_length=30)
################################ FIFO 30 items window for link th Batches (Stateful)##################################

cnt=1
pending_batch=[]
my_watch=[]


# Spark session
spark = SparkSession.builder \
    .appName("Hdfs_online_training_Stream") \
    .getOrCreate()

#  Read   temperatures stream from the "Training_nearRealTime_Stream" directory in HDFS
df = spark.readStream \
    .format("text") \
    .load("hdfs://127.0.0.1:9000/Training_nearRealTime_Stream/")
    # .load("hdfs://127.0.0.1:9000/Training_Stream/")







from keras.models import load_model
model_source = load_model('19_11_train_test_20_20_epoch_25.h5')
source_weights = model_source.get_weights()

model = Sequential()
model.add(GRU(100, activation='tanh', input_shape=(30, 2)))
model.add(Dense(1))
model.set_weights(source_weights)
model.compile(optimizer= 'adam', loss= 'mse' )
model.set_weights(source_weights)



rows=[]
cache_dataset=[]
iscasheExist=0


def process_data(batch_df, batch_id):
    global cnt
    global out
    global my_watch
    global cache_dataset
    global iscasheExist
    global  rows
    global  model



    curr_Batch=[]
    curr_Batch_date = []
    curr_Batch_temps = []


    if fifo_window.is_empty() ==False:
        rows = []
        for index, row in fifo_window.queue.iterrows():
            print(f'{row["data"]}')
            curr_Batch_temps.append(row["data"])
            curr_Batch_date.append("2015")
            rows.append(f'{row["data"]}')




    for row in batch_df.collect():
        curr_Batch_temps.append(row[0].split(',')[1].split('"')[0])
        curr_Batch_date.append(row[1].split(',')[0].split('"')[0])
        fifo_window.enqueue(row[0].split(',')[1].split('"')[0])  # last 7 days

    my_dataset_1 = pd.DataFrame(curr_Batch_date)  # Convert numpy array into pandas dataframe
    my_dataset_2 = pd.DataFrame(curr_Batch_temps)  # Convert numpy array into pandas dataframe

    my_dataset_2 = my_dataset_2.astype("float32")

    curr_Batch = pd.concat([my_dataset_1, my_dataset_2], axis=1)
    curr_Batch.columns = ['date', 'meantemp']



    # Split the data into training and testing sets
    train_size = int(len(curr_Batch) * 1.0)
    dl_train, dl_test = curr_Batch.iloc[:train_size], curr_Batch.iloc[train_size:]
    print(len(dl_train), len(dl_test))

    from sklearn.preprocessing import MinMaxScaler


    target_transformer = MinMaxScaler()

    dl_train['meantemp_diff'] = dl_train['meantemp'].diff().fillna(0)
    # dl_test['meantemp_diff'] = dl_test['meantemp'].diff().fillna(0)

    dl_train['meantemp'] = target_transformer.fit_transform(dl_train[['meantemp']])  # target
    # dl_test['meantemp'] = target_transformer.fit_transform(dl_test[['meantemp']])


    dl_train['meantemp_diff'] = target_transformer.fit_transform(dl_train[['meantemp_diff']])  # target
    # dl_test['meantemp_diff'] = target_transformer.fit_transform(dl_test[['meantemp_diff']])


    sequence_length = 30
    X_train, y_train = Generate_dataset(dl_train[['meantemp', 'meantemp_diff']], dl_train[['meantemp']], sequence_length)
    # X_test, y_test = Generate_dataset(dl_test[['meantemp', 'meantemp_diff']], dl_test[['meantemp']], sequence_length)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))

    print("X_train shape 3D : ", X_train.shape, " y_train shape : ", y_train.shape)

    if X_train.shape[0] <= 0:
        return



    history = model.fit(X_train, y_train, epochs=20, batch_size=1)

  #-------------------Arrange items by timestamp inside each bach---------------
    # timestamp_col = []
    # meantemp_col = []
    #
    # for row in batch_df.collect():
    #     timestamp_col.append(row[0].split(',')[0].split('"')[1])
    #     meantemp_col.append(row[0].split(',')[1].split('"')[0])
    #     # my_watch.append(row[0].split(',')[1].split('"')[0])
    #
    #
    # # if len(total)>0:
    # #     for row in total:
    # #         curr_Batch.append(row[0])
    # #     print("total is:",total)
    # #     total.clear()
    #
    #
    # # Combine the two lists into a list of tuples
    # data = list(zip(timestamp_col, meantemp_col))
    #
    # # Define the column names
    # columns = ['Timestamp', 'Meantemp']
    #
    # # Create DataFrame from the lists
    # df_temp = spark.createDataFrame(data, columns)
    #
    # df_temp_sorted_asc = df_temp.orderBy("Timestamp")
    #
    # # Copy 'Letter' column into a new column 'Letter_copy'
    # curr_Batch1 = df_temp_sorted_asc.select('Meantemp')
    #
    # for row in curr_Batch1.collect():
    #     curr_Batch.append(row[0])
    #
    # curr_Batch = pd.DataFrame(curr_Batch)
    # -------------------Arrange items by timestamp inside each bach---------------

    # print("curr_Batch length: ", len(curr_Batch))
    # print("fifo_window lengthh: ", fifo_window.count())
    # # fifo_window.print_elements()
    # print("=================")
    #
    # dataset_n = curr_Batch
    #
    # # dataset_n= pd.concat(fifo_window,curr_Batch)
    #
    #
    # # ensure all data is float
    # dataset_n = dataset_n.astype("float32")
    # values_n = dataset_n.values
    #
    #
    #
    #
    # # normalization
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_n = scaler.fit_transform(values_n)
    # # scaled_n = values_n
    #
    # i_in = 30  # past 30 temeratures
    # n_out = 1  # 31th day temperature
    # reframed_n = series_to_supervised(scaled_n, i_in, n_out)
    # values_spl_n = reframed_n.values
    #
    # train_size = int(len(values_spl_n))
    # train_n = values_spl_n[0:train_size, :]
    #
    # X_train_n, y_train_n = train_n[:, :-1], train_n[:, 30:31]
    # X_train_n = X_train_n.reshape((X_train_n.shape[0], X_train_n.shape[1] , 1))




    # # Online training -update our model with 'each' new data available-:
    # for t in range(y_train_n.shape[0]):
    #     x = X_train_n[t].reshape(1, 30, 1)  # a "new" input is available
    #     y = y_train_n[t].reshape(1, 1)
    #
    #     if out<1:
    #         model.fit(x, y, epochs=1, batch_size=8,
    #               callbacks=[lr_scheduler])
    #     out=out+1
    #     # model.train_on_batch(x, y)  # runs a single gradient update
    #     y_hat = (model.predict_on_batch(x))  # predict on the "new" input
    #
    #     print('-----', cnt, '-----')
    #     print('True label:      ', scaler.inverse_transform(y))
    #     print('Predicted label: ', scaler.inverse_transform(y_hat))
    #     cnt= cnt + 1

    print('model_ver'+str(cnt)+'_Saving...')
    model.save('Models/onlineModel_'+str(cnt)+'.h5')
    print("Saving is Done..")


    # saving the dataframe

    print("saving_RDD...")
    curr_Batch.to_csv('Batches/curr_Batch_'+str(cnt)+'.csv')
    print("saving_RDD Done")

    near_RealTime_RMSE()


    cnt = cnt + 1



from pyspark.sql.functions import input_file_name
# Add a new column 'file_name' that contains the path of the file
df_with_file_name = df.withColumn("file_name", input_file_name())

# Filter out files based on their name (e.g., skip files with '.tmp' extension)
df_filtered = df_with_file_name.filter(~df_with_file_name["file_name"].rlike(".*\\.tmp$"))





query = df_filtered.writeStream \
    .outputMode("append") \
    .foreachBatch(process_data) \
    .trigger(processingTime="30 seconds") \
    .start()


query.awaitTermination()


