import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler


data_dict = pickle.load( open("./ud120-projects/final_project/final_project_dataset.pkl", "r") )

data_dict.pop("TOTAL", 0)

feature_1 = "salary"
feature_2 = "exercised_stock_options"

def findRescaledValue(feature, value):
    array = []
    for k, v in data_dict.iteritems():
        if v[feature] != "NaN":
            array.append(v[feature])

    np_array = np.array(array)
    fitted_array = MinMaxScaler().fit(np_array)

    return fitted_array.transform([value])

    # np_stock = np.array(stock)
    # stock_fitted = MinMaxScaler().fit(np_stock)
    #
    # print salary_fitted.transform([200000.])
    # print stock_fitted.transform([1000000.])

print findRescaledValue(feature_1, 200000.), findRescaledValue(feature_2, 1000000.)
