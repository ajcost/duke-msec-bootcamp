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


def calculate_cpi(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the Consumer Price Index (CPI) for each country, year, and month.

    This function assumes the data has been cleaned. We will assume equal household weights for the
    goods, corresponding to the UNIT_OF_MEASUREMENT. This means each household purchases
    1 liter of milk, 1 loaf of bread, and 1 carton of eggs. While simplistic, this assumption
    provides a straightforward starting point.

    According to the U.S. Bureau of Labor Statistics, the formula for the CPI is:
    > A consumer price index (CPI) is usually calculated as a weighted average of the price change
    of the goods and services covered by the index. The weights are meant to reflect the relative
    importance of the goods and services as measured by their shares in the total consumption of
    households. The weight attached to each good or service determines the impact that its price
    change will have on the overall index. The weights should be made publicly available in the
    interests of transparency and for the information of the users of the index.

    Args:
        df (pd.DataFrame): The cleaned price dataset.

    Returns:
        pd.DataFrame: The Consumer Price Index (CPI) for each country, year, and month.
    """
    # Ensure the data is sorted for consistent grouping
    df = df.sort_values(by=[cols.COUNTRY, cols.YEAR, cols.MONTH, cols.FOOD_ITEM])

    # Initialize an empty list to store CPI results
    cpi_results = []

    # Base year and month for comparison
    base_year = 2018
    base_month = 1

    # Calculate base year prices
    # Calculate base year basket cost
    base_prices = df[(df[cols.YEAR] == base_year) & (df[cols.MONTH] == base_month)]
    base_prices_grouped = (
        base_prices.groupby(cols.COUNTRY)
        .agg({cols.PRICE_IN_USD: "sum"})
        .rename(columns={cols.PRICE_IN_USD: "BASE_BASKET_COST"})
    )

    # Group by Country and Year-Month
    grouped = df.groupby([cols.COUNTRY, cols.YEAR, cols.MONTH])

    for (country, year, month), group in grouped:
        # Current basket cost
        current_basket_cost = group[cols.PRICE_IN_USD].sum()

        # Base basket cost for the current country
        if country in base_prices_grouped.index:
            base_basket_cost = base_prices_grouped.loc[country, "BASE_BASKET_COST"]
        else:
            continue  # Skip if no base basket cost is available for the country

        # Calculate CPI
        cpi = (current_basket_cost / base_basket_cost) * 100

        # Append result to the list
        cpi_results.append(
            {"COUNTRY": country, "YEAR": year, "MONTH": month, "CPI": cpi}
        )

    # Convert results to a DataFrame
    cpi_df = pd.DataFrame(cpi_results)
    return cpi_df


def calculate_cpi_average_individual_product(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the Consumer Price Index AIP metric for each country, year, and month.

    We define the Consumer Price Index Average Individual Product (CPI AIP) as the average of the
    CPI for each individual product and then averaging these CPIs. This method allows us to
    equally weight the price effects of each of the individual products. Unlike the traditional CPI,
    which uses the fixed basket of goods, we had to make some assumptions about the household
    consumption of each product. This method just shows the average price change across all products.

    Args:
        df (pd.DataFrame): The cleaned price dataset.

    Returns:
        pd.DataFrame: The Consumer Price Index Average Individual Product (CPI AIP) for each country, year, and month.
    """
    df = df.sort_values(by=[cols.COUNTRY, cols.YEAR, cols.MONTH, cols.FOOD_ITEM])

    # Initialize an empty list to store CPI AIP results
    cpi_aip_results = []

    # Base year and month for comparison
    base_year = 2018
    base_month = 1

    # Calculate base year prices for each food item
    base_prices = df[(df[cols.YEAR] == base_year) & (df[cols.MONTH] == base_month)]
    base_prices_grouped = (
        base_prices.groupby([cols.COUNTRY, cols.FOOD_ITEM])[cols.PRICE_IN_USD]
        .mean()
        .reset_index()
    )
    base_prices_grouped = base_prices_grouped.rename(
        columns={cols.PRICE_IN_USD: "BASE_PRICE"}
    )

    # Merge base prices with the main dataframe
    df = df.merge(
        base_prices_grouped,
        on=[cols.COUNTRY, cols.FOOD_ITEM],
        how="left",
        suffixes=("", "_BASE"),
    )

    # Calculate CPI for each food item
    df["CPI"] = (df[cols.PRICE_IN_USD] / df["BASE_PRICE"]) * 100

    # Group by Country and Year-Month
    grouped = df.groupby([cols.COUNTRY, cols.YEAR, cols.MONTH])

    for (country, year, month), group in grouped:
        # Average CPI for all food items in the current group
        average_cpi = group["CPI"].mean()

        # Append result to the list
        cpi_aip_results.append(
            {"COUNTRY": country, "YEAR": year, "MONTH": month, "CPI_AIP": average_cpi}
        )

    # Convert results to a DataFrame
    cpi_aip_df = pd.DataFrame(cpi_aip_results)
    return cpi_aip_df


def calculate_cpi_product(
    df: pd.DataFrame,
    price_col: str = cols.PRICE_IN_USD,
) -> pd.DataFrame:
    """Calculates the Consumer Price Index (CPI) for each country, year, and month for each food item.

    This function will calculate the CPI or price index for each individual product or food item
    within the dataset. This function assumes that the data has been cleaned.

    Args:
        df (pd.DataFrame): The cleaned price dataset.
        price_col (str): The column name for the price data. Optional, defaults to "PRICE_IN_USD".

    Returns:
        pd.DataFrame: The Consumer Price Index (CPI) for each country, year, and month for each food item.
    """
    if price_col not in df.columns:
        raise ValueError(f"Price column {price_col} not found in DataFrame.")

    df = df.sort_values(by=[cols.COUNTRY, cols.YEAR, cols.MONTH, cols.FOOD_ITEM])

    # Initialize an empty list to store CPI results
    cpi_results = []

    # Base year and month for comparison
    base_year = 2018
    base_month = 1

    # Calculate base year prices for each food item
    base_prices = df[(df[cols.YEAR] == base_year) & (df[cols.MONTH] == base_month)]
    base_prices_grouped = (
        base_prices.groupby([cols.COUNTRY, cols.FOOD_ITEM])[price_col]
        .mean()
        .reset_index()
    )
    base_prices_grouped = base_prices_grouped.rename(columns={price_col: "BASE_PRICE"})

    # Merge base prices with the main dataframe
    df = df.merge(
        base_prices_grouped,
        on=[cols.COUNTRY, cols.FOOD_ITEM],
        how="left",
        suffixes=("", "_BASE"),
    )

    # Calculate CPI for each food item
    df["CPI"] = (df[price_col] / df["BASE_PRICE"]) * 100

    # Group by Country, Year-Month, and Food Item
    grouped = df.groupby([cols.COUNTRY, cols.YEAR, cols.MONTH, cols.FOOD_ITEM])

    for (country, year, month, food_item), group in grouped:
        # Current CPI for the food item
        cpi = group["CPI"].mean()

        # Append result to the list
        cpi_results.append(
            {
                "COUNTRY": country,
                "YEAR": year,
                "MONTH": month,
                "FOOD_ITEM": food_item,
                "CPI": cpi,
            }
        )

    # Convert results to a DataFrame
    cpi_df = pd.DataFrame(cpi_results)
    return cpi_df


def compare_cpi_difference(
    cpi_product_usd: pd.DataFrame,
    cpi_product_local: pd.DataFrame,
) -> pd.DataFrame:
    """Compares the CPI in USD and local currency, calculating the difference and returning the sorted values.

    Args:
        cpi_product_usd (pd.DataFrame): CPI data in USD.
        cpi_product_local (pd.DataFrame): CPI data in local currency.

    Returns:
        pd.DataFrame: DataFrame containing the CPI differences sorted in descending order.
    """
    # Rename CPI columns
    cpi_product_usd = cpi_product_usd.rename(columns={"CPI": "CPI_USD"})
    cpi_product_local = cpi_product_local.rename(columns={"CPI": "CPI_LOCAL"})

    # Merge the two dataframes on the relevant columns
    compare_cpis = cpi_product_usd.merge(
        cpi_product_local, on=["COUNTRY", "YEAR", "MONTH", "FOOD_ITEM"]
    )

    # Calculate the absolute difference in CPI
    compare_cpis["CPI_DIFFERENCE"] = abs(
        (compare_cpis.CPI_USD - compare_cpis.CPI_LOCAL) / compare_cpis.CPI_LOCAL
    )

    sorted_cpis = compare_cpis.sort_values(by="CPI_DIFFERENCE", ascending=False)

    return sorted_cpis
