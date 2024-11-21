import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf




dataframe1 = pd.read_csv("../Data/DailyDelhiClimateTrain.csv",parse_dates=['date'], index_col="date")

#====================ACF for Native Mean temperature time series=========================
dataframe1 = dataframe1[['meantemp']]
#====================ACF After Differencing=========================
dataframe1['meantemp_Differencing'] = dataframe1['meantemp'].diff().fillna(0)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
plot_acf(dataframe1['meantemp'], ax=axes[0], lags=100, title='ACF for native Meantemp')
plot_acf(dataframe1['meantemp_Differencing'], ax=axes[1], lags=100, title='ACF for Meantemp After Differencing')
plt.show()



