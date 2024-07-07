import os
import sys
import pandas as pd
import numpy as np

# setup src path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

# Load the data
import utils
import descriptive_statistics

utils.load_env(path="../.env")

# Load the data
data_path = f"{os.getenv("LOCAL_DATA_PATH")}/{os.getenv("FOOD_PRICES_FILENAME")}"


df = pd.read_csv(data_path)

df["Country"].unique()
df["Food Item"].unique()
df["Quality"].unique()
df["Currency"].unique()
df["Unit of Measurement"].unique()
df["Year"].unique()

# Check all years have all months
df.groupby("Year")["Month"].nunique()

df["Availability"]

# see if all countries have all food items
df.groupby("Country")["Food Item"].nunique()

# see if all countries have different qualities or there is a pattern
df.groupby("Country")["Quality"].nunique()

# which countries have the medium quality foods
df[df["Quality"] == "Medium"]["Country"].unique()

# create a function that for each country and food creates a 
# matrix of average price for each year
def create_price_matrix(df, country, food):
    df_country_food = df[(df["Country"] == country) & (df["Food Item"] == food)]
    years = df_country_food["Year"].unique()
    months = df_country_food["Month"].unique()
    price_matrix = np.zeros((len(years), len(months)))
    for i, year in enumerate(years):
        for j, month in enumerate(months):
            price_matrix[i, j] = df_country_food[(df_country_food["Year"] == year) & (df_country_food["Month"] == month)]["Price in USD"].mean()
    # convert to dataframe
    price_matrix = pd.DataFrame(price_matrix, columns=months, index=years)
    
    return price_matrix


create_price_matrix(df, "South Africa", "Bread")

# check missing values in any of the columns
df.isnull().sum()

df

df = utils.clean_data(df)

import pandas as pd
import numpy as np
from utils import *


descriptive_statistics.price_statistics(df, "South Africa", "Bread", "USD")

descriptive_statistics.inter_country_price_statistic_table(df, "Bread", "AVERAGE_PRICE")



