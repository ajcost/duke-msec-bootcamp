import pandas as pd
import numpy as np
from utils import *
import pandas as pd


def price_statistics(
    df: pd.DataFrame,
    country: str,
    food: str,
    currency: str,
) -> pd.DataFrame:
    """For the given country, food, and currency, create a matrix of average price,
    CPI between the 1st and 12th month, the greatest rolling CPI for the full year period,
    the total standard deviation, and variation in price for all years.

    Args:
        df (pd.DataFrame): The dataframe to use. Assumed it has been cleaned.
        country (str): The country to filter by.
        food (str): The food item to filter by.
        currency (str): The currency to filter by, can be "USD" or "local".

    Returns:
        pd.DataFrame: A matrix of average price and other statistics for each year.
    """
    if currency == "USD":
        price_column = cols.PRICE_IN_USD
    else:
        price_column = cols.AVERAGE_PRICE

    # Filter dataframe by country and food
    df_filtered = df[(df[cols.COUNTRY] == country) & (df[cols.FOOD_ITEM] == food)]

    # Initialize list to store results
    results_list = []

    # Group by year and calculate statistics for each year
    grouped = df_filtered.groupby(cols.YEAR)
    for year, group in grouped:
        # Calculate average price for the year
        avg_price = group[price_column].mean()

        # Calculate CPI between the 1st and 12th month
        cpi_start = group[group[cols.MONTH] == 1][price_column].values[0]
        cpi_end = group[group[cols.MONTH] == 12][price_column].values[0]
        cpi_change = (cpi_end - cpi_start) / cpi_start * 100

        # Calculate the greatest inter-month change
        group["Price_Diff"] = group[price_column].diff()
        max_inter_month_change = group["Price_Diff"].abs().max()

        # Calculate the greatest overall change between any two selected months
        max_overall_change = 0
        for i in range(1, 13):
            for j in range(i + 1, 13):
                cpi_i = group[group[cols.MONTH] == i][price_column].values[0]
                cpi_j = group[group[cols.MONTH] == j][price_column].values[0]
                overall_change = cpi_j - cpi_i
                if overall_change > max_overall_change:
                    max_overall_change = overall_change

        # Calculate total standard deviation of price
        std_dev = group[price_column].std()

        # Calculate total variation of price
        variance = group[price_column].var()

        # Append results to list
        results_list.append(
            {
                cols.YEAR: year,
                "AVERAGE_PRICE": avg_price,
                "PERC_PRICE_CHANGE_START_YEAR_TO_END_YEAR": cpi_change,
                "MAX_INTER_MONTH_PRICE_CHANGE": max_inter_month_change,
                "MAX_OVERALL_PRICE_CHANGE": max_overall_change,
                "PRICE_STANDARD_DEVIATION": std_dev,
                "PRICE_VARIANCE": variance,
            }
        )

    # Convert results list to DataFrame
    results_df = pd.DataFrame(results_list)

    # Set year as the index
    results_df.set_index(cols.YEAR, inplace=True)

    return results_df


def _average_price(group: pd.DataFrame) -> float:
    return group[cols.PRICE_IN_USD].mean()


def _perc_price_change(group: pd.DataFrame) -> float:
    cpi_start = group[group[cols.MONTH] == 1][cols.PRICE_IN_USD].values[0]
    cpi_end = group[group[cols.MONTH] == 12][cols.PRICE_IN_USD].values[0]
    return (cpi_end - cpi_start) / cpi_start * 100


def _max_inter_month_price_change(group: pd.DataFrame) -> float:
    group["Price_Diff"] = group[cols.PRICE_IN_USD].diff()
    return group["Price_Diff"].abs().max()


def _max_overall_price_change(group: pd.DataFrame) -> float:
    max_overall_change = 0
    for i in range(1, 13):
        for j in range(i + 1, 13):
            cpi_i = group[group[cols.MONTH] == i][cols.PRICE_IN_USD].values[0]
            cpi_j = group[group[cols.MONTH] == j][cols.PRICE_IN_USD].values[0]
            overall_change = abs(cpi_j - cpi_i)
            if overall_change > max_overall_change:
                max_overall_change = overall_change
    return max_overall_change


def _price_standard_deviation(group: pd.DataFrame) -> float:
    return group[cols.PRICE_IN_USD].std()


def _price_variance(group: pd.DataFrame) -> float:
    return group[cols.PRICE_IN_USD].var()


def _calculate_statistic(group: pd.DataFrame, statistic: str) -> float:
    if statistic == "AVERAGE_PRICE":
        return _average_price(group)
    elif statistic == "PERC_PRICE_CHANGE_START_YEAR_TO_END_YEAR":
        return _perc_price_change(group)
    elif statistic == "MAX_INTER_MONTH_PRICE_CHANGE":
        return _max_inter_month_price_change(group)
    elif statistic == "MAX_OVERALL_PRICE_CHANGE":
        return _max_overall_price_change(group)
    elif statistic == "PRICE_STANDARD_DEVIATION":
        return _price_standard_deviation(group)
    elif statistic == "PRICE_VARIANCE":
        return _price_variance(group)
    else:
        raise ValueError(f"Unsupported statistic: {statistic}")


def inter_country_price_statistic_table(
    df: pd.DataFrame,
    food_item: str,
    statistic: str,
) -> pd.DataFrame:
    """Create a table for a given food item and statistic, comparing the statistic across all countries, and years.

    The statistic should be one of the following: "AVERAGE_PRICE", "PERC_PRICE_CHANGE_START_YEAR_TO_END_YEAR",
    "MAX_INTER_MONTH_PRICE_CHANGE", "MAX_OVERALL_PRICE_CHANGE", "PRICE_STANDARD_DEVIATION",
    "PRICE_VARIANCE".

    This function assumes the data has been cleaned.

    Args:
        food_item (str): The food item to filter by.
        statistic (str): The statistic to compare across countries and years.
    """
    df_filtered = df[df[cols.FOOD_ITEM] == food_item]
    results_list = []
    grouped = df_filtered.groupby([cols.COUNTRY, cols.YEAR])

    for (country, year), group in grouped:
        statistic_value = _calculate_statistic(group, statistic)
        results_list.append(
            {
                cols.COUNTRY: country,
                cols.YEAR: year,
                statistic: statistic_value,
            }
        )

    results_df = pd.DataFrame(results_list)
    results_pivot = results_df.pivot(
        index=cols.YEAR,
        columns=cols.COUNTRY,
        values=statistic,
    )

    # Calculate the average value for each year across all countries and months
    yearly_averages = df_filtered.groupby(cols.YEAR)[cols.PRICE_IN_USD].mean()
    results_pivot["Average"] = yearly_averages

    return results_pivot


def descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Builds a description of the dataset using pandas describe function.

    First the unnecessary columns are removed and then the describe function is called.
    The dataframe is grouped by country year and food item. The returned df
    can be indexed by column and by row using the .loc method.

    Args:
        df (pd.DataFrame): The dataframe to describe.

    Returns:
        pd.DataFrame: The description of the dataset.
    """

    descriptions = (
        df.drop([cols.MONTH, cols.AVAILABILITY], axis=1)
        .groupby([cols.COUNTRY, cols.FOOD_ITEM, cols.YEAR])
        .describe()
    )

    return descriptions
