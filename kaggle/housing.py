import pandas as pd
from datetime import datetime


iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

avg_lot_size = round(pd.DataFrame.mean(home_data.get("LotArea")))

newest_home_age = datetime.now().year - pd.DataFrame.max(home_data.get("YearBuilt"))
