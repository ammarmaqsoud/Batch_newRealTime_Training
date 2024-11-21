import math

import subprocess
import subprocess

import os
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

os.environ['SPARK_HOME']="/opt/spark"

java8_location= '/usr/lib/jvm/java-8-openjdk-amd64' # Set your own
os.environ['JAVA_HOME'] = java8_location

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




# import tensorflow as tf
# new_model = tf.keras.models.load_model('delhi_model')
# new_model.summary()


import matplotlib
matplotlib.use('Agg')
import pandas
import datetime as dt

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.util.common import Sample
import matplotlib.pyplot as plt
from bigdl.dataset.transformer import *
from matplotlib.pyplot import imshow
from pyspark import SparkContext


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


import pandas as pd
import glob

combined_df = pd.DataFrame()

df = pd.read_csv("../Data/delhi_temp_from_hdfs/0.csv")
combined_df = pd.concat([combined_df, df])
df = pd.read_csv("../Data/delhi_temp_from_hdfs/1.csv")
combined_df = pd.concat([combined_df, df])
df = pd.read_csv("../Data/delhi_temp_from_hdfs/2.csv")
combined_df = pd.concat([combined_df, df])
df = pd.read_csv("../Data/delhi_temp_from_hdfs/3.csv")
combined_df = pd.concat([combined_df, df])
df = pd.read_csv("../Data/delhi_temp_from_hdfs/4.csv")
combined_df = pd.concat([combined_df, df])
df = pd.read_csv("../Data/delhi_temp_from_hdfs/5.csv")
combined_df = pd.concat([combined_df, df])
df = pd.read_csv("../Data/delhi_temp_from_hdfs/6.csv")
combined_df = pd.concat([combined_df, df])


print((combined_df.count(), len(combined_df.columns)))


temperatures = []
floatTemperatures = np.array(temperatures, dtype = np.float32)

 # Using pandas
import pandas as pd

i=0
pandasDF = combined_df#.toPandas()
for index, row in pandasDF.iterrows():
    if i==0:#bypass column name
        i=i+1
        continue
    # print(row['_c0'])
    # floatTemperatures = np.append(floatTemperatures, row['_c0'].split(',')[1])
    floatTemperatures = np.append(floatTemperatures, row['_c0'])


my_dataset = pd.DataFrame(floatTemperatures)# Convert numpy array into pandas dataframe


my_dataset = my_dataset.astype("float32")
values     = my_dataset.values
#print("values : \n",values)



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# scaled = values
#print("scaled : \n",scaled)






def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
      n_vars = 1 if type(data) is list else data.shape[1]
      df = pd.DataFrame(data)
      cols, names = list(), list()
      for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [("var%d(t-%d)" % (j+1, i)) for j in range(n_vars)]
      for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                  names += [("var%d(t)" % (j+1)) for j in range(n_vars)]
            else:
                  names += [("var%d(t+%d)" % (j+1, i)) for j in range(n_vars)]
      agg = pd.concat(cols, axis=1)
      agg.columns = names
      if dropnan:
            agg.dropna(inplace=True)
      return agg


i_in  = 7 # past observations
n_out = 1 # future observations
reframed = series_to_supervised(scaled, i_in, n_out)
print(reframed.shape)

# print(reframed.loc[7])
# print("**************************")
#
# print(reframed.loc[7][0:7])
# print(reframed.loc[7][7])

def gen_rand_sample(index):
    features = np.array(reframed.loc[index][0:7])
    label = np.array(reframed.loc[index][7])
    # features = np.random.uniform(0, 1, (FEATURES_DIM))
    # label = (2 * features).sum() + 0.4
    return Sample.from_ndarray(features, label)





sc=SparkContext.getOrCreate(conf=create_spark_conf().setMaster("local[4]").set("spark.driver.memory","2g"))
init_engine()

rdd_train = sc.parallelize(range(7, 1461)).map( lambda i: gen_rand_sample(i) )

# Parameters
learning_rate = 0.01
training_epochs = 50
batch_size = 32



# rdd_train=sc.parallelize(values_spl)


def linear_regression(n_input, n_output):
    # Initialize a sequential container
    model1 = Sequential()
    # Add a linear layer
    model1.add(Linear(n_input, n_output))

    return model1

# def linear_regression1(n_input, n_output):
#     # Initialize a sequential container
#     model = Sequential()
#     recurrent = Recurrent()
#     recurrent.add(LSTM(7, 64))
#     # model.add(InferReshape([-1, 7], True))
#     # model.add(recurrent)
#     # model.add(Select(2, -1))
#     model.add(Linear(7, 1))

# def linear_regression1(n_input, n_output):
#     # Initialize a sequential container
#     model = Sequential()
#     recurrent = Recurrent()
#     recurrent.add(LSTM(7, 64))
#     model.add(InferReshape([-1, 7], True))
#     model.add(recurrent)
#     model.add(Select(2, -1))
#
#     model.add(Linear(64, 1))
#
#     return model


def linear_regression1(n_input, n_output):
    # Initialize a sequential container
    model = Sequential()
    recurrent = Recurrent()
    recurrent.add(GRU(7, 64))
    model.add(InferReshape([-1, 7], True))
    model.add(recurrent)
    model.add(Select(2, -1))

    model.add(Linear(64, 1))

    return model


model = linear_regression1(7, 1)

# Create an Optimizer
optimizer = Optimizer(
    model=model,
    training_rdd=rdd_train,
    criterion=MSECriterion(),
    optim_method=SGD(learningrate=learning_rate),
    end_trigger=MaxEpoch(training_epochs),
    batch_size=batch_size)



# Start to train
trained_model = optimizer.optimize()


# Print the first five predicted results of training data.
predict_result = trained_model.predict(rdd_train)
p = predict_result.take(5)



#=========================== Testing ========================================


combined_df = pd.read_csv("../Data/DailyDelhiClimateTest.csv")

temperatures = []
floatTemperatures = np.array(temperatures, dtype = np.float32)

 # Using pandas
import pandas as pd

i=0
pandasDF = combined_df#.toPandas()
for index, row in pandasDF.iterrows():
    # if i==0:#bypass column name
    #     i=i+1
    #     continue
    # print(row['_c0'])
    # floatTemperatures = np.append(floatTemperatures, row['_c0'].split(',')[1])
    floatTemperatures = np.append(floatTemperatures, row['meantemp'])
# floatTemperatures = np.append(floatTemperatures, 10.0)#add last temperature

my_dataset = pd.DataFrame(floatTemperatures)# Convert numpy array into pandas dataframe

# ensure all data is float
my_dataset = my_dataset.astype("float32")
values     = my_dataset.values
#print("values : \n",values)


# normalize features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
#print("scaled : \n",scaled)



# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
      n_vars = 1 if type(data) is list else data.shape[1]
      df = pd.DataFrame(data)
      cols, names = list(), list()
      for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [("var%d(t-%d)" % (j+1, i)) for j in range(n_vars)]
      for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                  names += [("var%d(t)" % (j+1)) for j in range(n_vars)]
            else:
                  names += [("var%d(t+%d)" % (j+1, i)) for j in range(n_vars)]
      agg = pd.concat(cols, axis=1)
      agg.columns = names
      if dropnan:
            agg.dropna(inplace=True)
      return agg


# reshape into X=t and Y=t+1
i_in  = 7 # past observations
n_out = 1 # future observations
reframed = series_to_supervised(scaled, i_in, n_out)
print(reframed.shape)



def gen_rand_sample(index):
    features = np.array(reframed.loc[index][0:7])
    label = np.array(reframed.loc[index][7])
    # features = np.random.uniform(0, 1, (FEATURES_DIM))
    # label = (2 * features).sum() + 0.4
    return Sample.from_ndarray(features, label)


predict_data = sc.parallelize(range(7, 107)).map( lambda i: gen_rand_sample(i) )




y_test_1= scaler.fit_transform([[15.6842105263158]])
y_test_2= scaler.fit_transform([[14.5714285714286]])
y_test_3= scaler.fit_transform([[12.1111111111111]])
y_test_4= scaler.fit_transform([[11]])
y_test_5= scaler.fit_transform([[11.7894736842105]])
y_test_6= scaler.fit_transform([[13.2352941176471]])
y_test_7 = scaler.fit_transform([[13.2]])
y_test_8 = scaler.fit_transform([[16.4347826086957]])
y_test_9 = scaler.fit_transform([[14.65]])
y_test_10 = scaler.fit_transform([[11.7222222222222]])
y_test_11 = scaler.fit_transform([[13.0416666666667]])
y_test_12 = scaler.fit_transform([[14.6190476190476]])
y_test_13 = scaler.fit_transform([[15.2631578947368]])
y_test_14= scaler.fit_transform([[15.3913043478261]])
y_test_15 = scaler.fit_transform([[18.44]])
y_test_16 = scaler.fit_transform([[18.1176470588235]])
y_test_17 = scaler.fit_transform([[18.3478260869565]])
y_test_18 = scaler.fit_transform([[21]])
y_test_19 = scaler.fit_transform([[16.1785714285714]])
y_test_20 = scaler.fit_transform([[16.5]])
y_test_21 = scaler.fit_transform([[14.8636363636364]])
y_test_22 = scaler.fit_transform([[15.6666666666667]])
y_test_23 = scaler.fit_transform([[16.4444444444444]])
y_test_24 = scaler.fit_transform([[16.125]])
y_test_25 = scaler.fit_transform([[15.25]])
y_test_26 = scaler.fit_transform([[17.0909090909091]])
y_test_27 = scaler.fit_transform([[15.6363636363636]])
y_test_28 = scaler.fit_transform([[18.7]])
y_test_29 = scaler.fit_transform([[18.6315789473684]])
y_test_30 = scaler.fit_transform([[16.8888888888889]])
y_test_31 = scaler.fit_transform([[15.125]])
y_test_32 = scaler.fit_transform([[15.7]])
y_test_33 = scaler.fit_transform([[15.375]])
y_test_34 = scaler.fit_transform([[14.6666666666667]])
y_test_35 = scaler.fit_transform([[15.625]])
y_test_36 = scaler.fit_transform([[16.25]])
y_test_37 = scaler.fit_transform([[16.3333333333333]])
y_test_38 = scaler.fit_transform([[16.875]])
y_test_39 = scaler.fit_transform([[17.5714285714286]])
y_test_40 = scaler.fit_transform([[20.25]])
y_test_41 = scaler.fit_transform([[21.3]])
y_test_42 = scaler.fit_transform([[21.125]])
y_test_43 = scaler.fit_transform([[22.3636363636364]])
y_test_44 = scaler.fit_transform([[23.375]])
y_test_45 = scaler.fit_transform([[21.8333333333333]])
y_test_46 = scaler.fit_transform([[19.125]])
y_test_47 = scaler.fit_transform([[18.625]])
y_test_48 = scaler.fit_transform([[19.125]])
y_test_49 = scaler.fit_transform([[19]])
y_test_50 = scaler.fit_transform([[18.75]])


y_test_51 = scaler.fit_transform([[19.875]])
y_test_52 = scaler.fit_transform([[23.3333333333333 ]])
y_test_53 = scaler.fit_transform([[ 24.4615384615385]])
y_test_54 = scaler.fit_transform([[ 23.75]])
y_test_55 = scaler.fit_transform([[ 20.5]])
y_test_56 = scaler.fit_transform([[ 19.125]])
y_test_57 = scaler.fit_transform([[ 19.75]])
y_test_58 = scaler.fit_transform([[ 20]])
y_test_59 = scaler.fit_transform([[ 22.625]])
y_test_60 = scaler.fit_transform([[ 21.5454545454545]])
y_test_61 = scaler.fit_transform([[ 20.7857142857143]])
y_test_62 = scaler.fit_transform([[ 19.9375]])
y_test_63 = scaler.fit_transform([[18.5333333333333 ]])
y_test_64 = scaler.fit_transform([[ 17.375]])
y_test_65 = scaler.fit_transform([[ 17.4444444444444]])
y_test_66 = scaler.fit_transform([[ 18]])
y_test_67 = scaler.fit_transform([[ 19.875]])
y_test_68 = scaler.fit_transform([[ 24]])
y_test_69 = scaler.fit_transform([[ 20.9]])
y_test_70 = scaler.fit_transform([[ 24.6923076923077]])
y_test_71 = scaler.fit_transform([[24.6666666666667 ]])
y_test_72 = scaler.fit_transform([[ 23.3333333333333]])
y_test_73 = scaler.fit_transform([[25 ]])
y_test_74 = scaler.fit_transform([[ 27.25]])
y_test_75 = scaler.fit_transform([[ 28]])
y_test_76 = scaler.fit_transform([[ 28.9166666666667]])
y_test_77 = scaler.fit_transform([[ 26.5]])
y_test_78 = scaler.fit_transform([[ 29.1]])
y_test_79 = scaler.fit_transform([[ 29.5]])
y_test_80 = scaler.fit_transform([[ 29.8888888888889]])
y_test_81 = scaler.fit_transform([[ 31]])
y_test_82 = scaler.fit_transform([[ 29.2857142857143]])
y_test_83 = scaler.fit_transform([[30.625 ]])
y_test_84 = scaler.fit_transform([[ 31.375]])
y_test_85 = scaler.fit_transform([[29.75 ]])
y_test_86 = scaler.fit_transform([[ 30.5]])
y_test_87 = scaler.fit_transform([[ 30.9333333333333]])
y_test_88 = scaler.fit_transform([[ 29.2307692307692]])
y_test_89 = scaler.fit_transform([[ 31.2222222222222]])
y_test_90 = scaler.fit_transform([[ 27]])
y_test_91 = scaler.fit_transform([[ 25.625]])
y_test_92 = scaler.fit_transform([[27.125 ]])
y_test_93 = scaler.fit_transform([[ 27.8571428571429]])
y_test_94 = scaler.fit_transform([[ 29.25]])
y_test_95 = scaler.fit_transform([[ 29.25]])
y_test_96 = scaler.fit_transform([[ 29.6666666666667]])
y_test_97 = scaler.fit_transform([[ 30.5]])
y_test_98 = scaler.fit_transform([[ 31.2222222222222]])
y_test_99 = scaler.fit_transform([[31 ]])
y_test_100 = scaler.fit_transform([[ 32.5555555555556]])
y_test_101 = scaler.fit_transform([[ 34]])
y_test_102 = scaler.fit_transform([[ 33.5]])
y_test_103 = scaler.fit_transform([[ 34.5]])
y_test_104 = scaler.fit_transform([[ 34.25]])
y_test_105 = scaler.fit_transform([[32.9 ]])
y_test_106 = scaler.fit_transform([[ 32.875]])
y_test_107 = scaler.fit_transform([[ 32]])














def test_predict(trained_model):
    predict_result = trained_model.predict(predict_data)
    p = predict_result.take(107)
    ground_label = np.array([y_test_1, y_test_2, y_test_3,y_test_4,y_test_5,y_test_6,y_test_7,y_test_8,y_test_9,y_test_10,y_test_11,y_test_12,y_test_13,y_test_14,y_test_15,y_test_16,y_test_17,y_test_18,y_test_19,y_test_20,y_test_21,y_test_22,y_test_23,y_test_24,y_test_25,y_test_26,y_test_27,y_test_28,y_test_29,y_test_30,y_test_31,y_test_32,y_test_33,y_test_34,y_test_35,y_test_36,y_test_37,y_test_38,y_test_39,y_test_40,y_test_41,y_test_42,y_test_43,y_test_44,y_test_45,y_test_46,y_test_47,y_test_48,y_test_49,y_test_50
                            ,y_test_51,y_test_52,y_test_53,y_test_54,y_test_55,y_test_56,y_test_57,y_test_58,y_test_59,y_test_60,y_test_61,y_test_62,y_test_63,y_test_64,y_test_65,y_test_66,y_test_67,y_test_68,y_test_69
                               , y_test_70, y_test_71, y_test_72, y_test_73, y_test_74, y_test_75, y_test_76, y_test_77, y_test_78, y_test_79, y_test_80, y_test_81, y_test_82, y_test_83, y_test_84, y_test_85, y_test_86, y_test_87, y_test_88, y_test_89
                               , y_test_90, y_test_91, y_test_92, y_test_93, y_test_94, y_test_95, y_test_96, y_test_97, y_test_98, y_test_99
                               , y_test_100, y_test_101, y_test_102, y_test_103, y_test_104, y_test_105, y_test_106, y_test_107
                             ], dtype="float32")
    RMSE = math.sqrt(((p - ground_label) ** 2).mean())
    print("RMSE: ",RMSE)


test_predict(trained_model)
