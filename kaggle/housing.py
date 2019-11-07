import pandas as pd
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

avg_lot_size = round(pd.DataFrame.mean(home_data.get("LotArea")))

newest_home_age = datetime.now().year - pd.DataFrame.max(home_data.get("YearBuilt"))

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data.columns

melbourne_data = melbourne_data.dropna(axis=0)

y = melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
X.describe()
X.head()

melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))


##### model validation #####

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# define model
melbourne_model = DecisionTreeRegressor()
# fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
