import pandas as pd
from descriptive_statistics import *
from utils import *


def calculate_exchange_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the exchange rate for each product, country, and year/month.

    Assumes dataset is cleaned.

    Args:
        df (pd.DataFrame): The dataframe to calculate the exchange rate for.

    Returns:
        pd.DataFrame: The dataframe with the exchange rate calculated.
    """
    df["EXCHANGE_RATE"] = df.PRICE_IN_USD / df.AVERAGE_PRICE

    return df


def check_exchange_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Checks that during the same year and month, the exchange rate is the same for all products within a country.

    Assumes dataset is cleaned.

    Args:
        df (pd.DataFrame): The dataframe to calculate the exchange rate for.

    Returns:
        bool: True if the exchange rate is the same for all products within a country, for that year and month. False otherwise.
    """
    df = calculate_exchange_rate(df)

    exchange_rate_check = df.groupby(["COUNTRY", "YEAR", "MONTH"])[
        "EXCHANGE_RATE"
    ].nunique()

    return exchange_rate_check.nunique() == 1


def calculate_average_exchange_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the average exchange rate for each country.

    Assumes dataset is cleaned.

    Args:
        df (pd.DataFrame): The dataframe to calculate the exchange rate for.

    Returns:
        pd.DataFrame: The dataframe with the average exchange rate calculated.
    """
    df = calculate_exchange_rate(df)

    df = df.groupby(["COUNTRY", "YEAR", "MONTH"])["EXCHANGE_RATE"].mean().reset_index()

    return df


def calculate_normalized_average_exchange_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the normalized average exchange rate for each country.

    Assumes dataset is cleaned.

    Args:
        df (pd.DataFrame): The dataframe to calculate the exchange rate for.

    Returns:
        pd.DataFrame: The dataframe with the normalized average exchange rate calculated.
    """
    df = calculate_average_exchange_rate(df)

    # need to divide the exchange rate for each country by the max exchange rate for that country
    df["NORMALIZED_EXCHANGE_RATE"] = df.groupby("COUNTRY")["EXCHANGE_RATE"].transform(
        lambda x: x / x.max()
    )

    # subtract the mean of the normalized exchange rate for each country
    # This is called mean centering
    df["NORMALIZED_EXCHANGE_RATE"] = df.groupby("COUNTRY")[
        "NORMALIZED_EXCHANGE_RATE"
    ].transform(lambda x: x - x.mean())

    return df


def find_all_arbitrage_opportunities(df: pd.DataFrame, year=None) -> pd.DataFrame:
    """Finds all arbitrage opportunities for a given year.

    Assumes dataset is cleaned.

    Args:
        df (pd.DataFrame): The dataframe to calculate the exchange rate for.
        year (int): The year to find arbitrage opportunities for.

    Returns:
        pd.DataFrame: The dataframe with the arbitrage opportunities.
    """
    if year:
        df = df[df.YEAR == year]

    select_cols = [
        cols.COUNTRY,
        cols.YEAR,
        cols.MONTH,
        cols.FOOD_ITEM,
        cols.PRICE_IN_USD,
    ]

    join_cols = [cols.YEAR, cols.MONTH, cols.FOOD_ITEM]

    arbitrage_opportunities = pd.merge(
        df[select_cols],
        df[select_cols],
        on=join_cols,
        how="outer",
    )

    arbitrage_opportunities = arbitrage_opportunities[
        arbitrage_opportunities.PRICE_IN_USD_x < arbitrage_opportunities.PRICE_IN_USD_y
    ]

    # rename x and y to buy and sell
    arbitrage_opportunities = arbitrage_opportunities.rename(
        columns={
            cols.COUNTRY + "_x": "BUY_COUNTRY",
            cols.COUNTRY + "_y": "SELL_COUNTRY",
            cols.PRICE_IN_USD + "_x": "BUY_PRICE",
            cols.PRICE_IN_USD + "_y": "SELL_PRICE",
        }
    )

    arbitrage_opportunities["PROFIT"] = (
        arbitrage_opportunities.SELL_PRICE - arbitrage_opportunities.BUY_PRICE
    )

    return arbitrage_opportunities


def greatest_arbitrage_opportunities(
    df: pd.DataFrame,
    original_df: pd.DataFrame = None,
    cpi_adjusted: bool = False,
) -> pd.DataFrame:
    """Finds the greatest arbitrage opportunities for each year, month, product group.

    If the `cpi_adjusted` parameter is set to True, the profit will be adjusted for inflation, the
    original price dataframe then must be provided. If it is not provided, the function will return
    an error.

    Assumes the original dataframe is cleaned.

    Args:
        df (pd.DataFrame): The dataframe that is returned from `find_all_arbitrage_opportunities`.
        original_df (pd.DataFrame): The original dataframe with prices. Must be provided if `cpi_adjusted` is True.
        cpi_adjusted (bool): Whether to adjust the profit for inflation.

    Returns:
        pd.DataFrame: The dataframe with the greatest arbitrage opportunities.
    """
    greatest_arbitrage_opportunities = (
        df.groupby([cols.YEAR, cols.MONTH, cols.FOOD_ITEM])["PROFIT"]
        .max()
        .reset_index()
    )

    if cpi_adjusted:
        cpi = calculate_cpi(original_df)

        cpi = cpi.groupby([cols.YEAR, cols.MONTH])["CPI"].mean().reset_index()

        greatest_arbitrage_opportunities = pd.merge(
            greatest_arbitrage_opportunities,
            cpi,
            on=[cols.YEAR, cols.MONTH],
            how="left",
        )

        greatest_arbitrage_opportunities["PROFIT"] = greatest_arbitrage_opportunities[
            "PROFIT"
        ] / (
            greatest_arbitrage_opportunities["CPI"] / 100
        )  # have to divide by 100 since CPI is in percentage

    return greatest_arbitrage_opportunities


def greatest_country_to_country_arbitrage_opportunities(
    arbitrage_opportunities: pd.DataFrame, df: pd.DataFrame
) -> pd.DataFrame:
    """Finds the greatest arbitrage opportunities for each country to country pair.

    Assumes dataset is cleaned. Automatically adjusts for inflation.

    Args:
        arbitrage_opportunities (pd.DataFrame): The dataframe that is returned from `find_all_arbitrage_opportunities`.
        df (pd.DataFrame): The original dataframe with prices.

    Returns:
        pd.DataFrame: The dataframe with the greatest arbitrage country to country opportunities.
    """
    greatest_arbitrage_opportunities_df = greatest_arbitrage_opportunities(
        arbitrage_opportunities, df, cpi_adjusted=True
    )

    # merge CPI with artbitrage opportunities
    arbitrage_opportunities = pd.merge(
        arbitrage_opportunities,
        greatest_arbitrage_opportunities_df[
            [cols.YEAR, cols.MONTH, cols.FOOD_ITEM, "CPI"]
        ],
        on=[cols.YEAR, cols.MONTH, cols.FOOD_ITEM],
        how="left",
    )

    # calculate the profit adjusted for inflation
    arbitrage_opportunities["CPI_ADJUSTED_PROFIT"] = arbitrage_opportunities[
        "PROFIT"
    ] / (arbitrage_opportunities["CPI"] / 100)

    # groupby country to country pair and find the average cpi adjusted profit
    # and calculate the average percentage profit
    greatest_country_to_country_arbitrage_opportunities_df = (
        arbitrage_opportunities.groupby(
            ["BUY_COUNTRY", "SELL_COUNTRY", "FOOD_ITEM", "CPI"]
        )
        .agg(
            {
                "CPI_ADJUSTED_PROFIT": "mean",
                "BUY_PRICE": "mean",
            }
        )
        .reset_index()
    )

    greatest_country_to_country_arbitrage_opportunities_df[
        "PROFIT_RATIO"
    ] = greatest_country_to_country_arbitrage_opportunities_df[
        "CPI_ADJUSTED_PROFIT"
    ] / (
        greatest_country_to_country_arbitrage_opportunities_df["BUY_PRICE"]
        / (greatest_country_to_country_arbitrage_opportunities_df["CPI"] / 100)
    )

    # sort by percentage profit
    greatest_country_to_country_arbitrage_opportunities_df = (
        greatest_country_to_country_arbitrage_opportunities_df.groupby(
            ["BUY_COUNTRY", "SELL_COUNTRY", "FOOD_ITEM"]
        )
        .agg(
            {
                "CPI_ADJUSTED_PROFIT": "mean",
                "PROFIT_RATIO": "mean",
            }
        )
        .reset_index()
    ).sort_values("PROFIT_RATIO", ascending=False)

    return greatest_country_to_country_arbitrage_opportunities_df
