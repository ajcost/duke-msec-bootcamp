from typing import Union
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

from descriptive_statistics import *
from arbitrage import *
from utils import *


def normalize_to_cpi(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes the price for each country, product year to the
    price index for each country, product, and year.

    This does this in a no-leakage way by calculate the normalized price based on
    the previous month's price-index.

    Args:
        df (pd.DataFrame): The dataframe to normalize the exchange rate for.

    Returns:
        pd.DataFrame: The dataframe with the exchange rate normalized to the CPI.
    """
    cpi_index = calculate_cpi_product(df)

    # join the cpi index to the original data
    df = df.merge(cpi_index, on=["COUNTRY", "YEAR", "MONTH", "FOOD_ITEM"])

    df["CPI_DIFFERENCE"] = df.groupby(["COUNTRY", "FOOD_ITEM"])["CPI"].diff()

    return df


def check_stationarity_of_timeseries(
    df: pd.DataFrame, country: str, food_item: str, col: str
) -> bool:
    """Checks if the time series of the price of a product in a country is stationary.

    This assumes the dataset is cleaned and has a column for the CPI_ADJUSTED_PRICE.

    Args:
        df (pd.DataFrame): The dataframe to check the stationarity for.
        country (str): The country to check the stationarity for.
        food_item (str): The product to check the stationarity for.
        col (str): The column to check the stationarity for.

    Returns:
        bool: True if the time series of the price of a product in a country is stationary. False otherwise.
    """
    # Filter the dataframe for the specific country and product and year 2022 due to price shock
    filtered_df = df[
        (df["COUNTRY"] == country)
        & (df["FOOD_ITEM"] == food_item)
        & (df["YEAR"] != 2022)
    ]

    # Extract the CPI_ADJUSTED_PRICE series
    time_series = filtered_df[col].dropna()

    # Perform the Augmented Dickey-Fuller test
    result = adfuller(time_series)

    # Extract the p-value from the test result
    p_value = result[1]

    # If the p-value is less than 0.05, we reject the null hypothesis and conclude the time series is stationary
    return p_value < 0.05, p_value


def build_vector_autoregression(
    df: pd.DataFrame,
    col: str,
    country: str = None,
    food_item: str = None,
):
    """Builds a Vector Autoregression (VAR) model for the given column, assuming the time series
    data is stationary at the country and product level.

    Either one of country or food_item needs to be provided and data is filtered for it. If country is provided
    the data is filtered for country. If food_item is provided then data is filtered for food_item. Otherwise
    an error is thrown.

    Args:
        df (pd.DataFrame): The dataframe to build the VAR model for.
        col (str): The column to build the VAR model for.
        country (str, optional): The country to filter the data for.
        food_item (str, optional): The food item to filter the data for.

    Returns:
        VARResults: The fitted VAR model.
    """
    if not country and not food_item:
        raise ValueError("Either 'country' or 'food_item' must be provided")

    # Filter relevant columns
    relevant_columns = ["COUNTRY", "FOOD_ITEM", "YEAR", "MONTH", col]
    df = df[relevant_columns]

    # Filter 2022 data
    df = df[df.YEAR != 2022]

    # Apply country or food_item filter
    if country:
        df = df[df.COUNTRY == country]
    if food_item:
        df = df[df.FOOD_ITEM == food_item]

    # Handle missing values by interpolation
    df[col] = df[col].interpolate(method="linear")

    # Ensure the data is sorted
    df = df.sort_values(by=["COUNTRY", "FOOD_ITEM", "YEAR", "MONTH"])

    # Prepare the data for VAR model
    var_data = df.pivot_table(
        index=["YEAR", "MONTH"], columns=["COUNTRY", "FOOD_ITEM"], values=col
    )
    var_data = var_data.dropna()

    # Fit the VAR model
    model = VAR(var_data)
    fitted_model = model.fit(
        maxlags=2, ic="aic"
    )  # Use Akaike Information Criterion to select the best lag

    return fitted_model


def concise_var_summary(fitted_model):
    """Generates a concise summary of the fitted VAR model.

    Args:
        fitted_model (VARResults): The fitted VAR model.

    Returns:
        str: A concise summary of the VAR model.
    """
    summary = []

    # Model metrics
    summary.append(f"Model: VAR")
    summary.append(f"Number of Equations: {fitted_model.neqs}")
    summary.append(f"Number of Observations: {fitted_model.nobs}")
    summary.append(f"AIC: {fitted_model.aic:.4f}")
    summary.append(f"BIC: {fitted_model.bic:.4f}")
    summary.append(f"HQIC: {fitted_model.hqic:.4f}")
    summary.append(f"FPE: {fitted_model.fpe:.4e}")
    summary.append(f"Log Likelihood: {fitted_model.llf:.4f}")

    summary.append("\nSignificant Coefficients:")

    # Coefficients and their significance
    for equation in fitted_model.params.columns:
        summary.append(f"\nResults for equation {equation}:")
        coeffs = fitted_model.params[equation]
        pvalues = fitted_model.pvalues[equation]
        for term in coeffs.index:
            coef = coeffs[term]
            pval = pvalues[term]
            if pval < 0.05:
                summary.append(
                    f"  {term}: coefficient = {coef:.4f}, p-value = {pval:.4f}"
                )

    return "\n".join(summary)


def load_data(df: pd.DataFrame, food_item: str) -> pd.DataFrame:
    """This function filters the dataframe for a specific food item and pivots the data for prediction.

    This function assumes that the previous preprocessing steps have been completed.

    Args:
        df (pd.DataFrame): The dataframe to filter.
        food_item (str): The food item to filter the dataframe for.
    """
    df = df[df["FOOD_ITEM"] == food_item].copy()

    df = df.pivot_table(
        index=["YEAR", "MONTH"],
        columns="COUNTRY",
        values="DIFF",
    ).dropna()

    return df


def normalize_data(df_pivot: pd.DataFrame) -> pd.DataFrame:
    """Normalizes the data using MinMaxScaler.

    Args:
        df_pivot (pd.DataFrame): The dataframe to normalize.

    Returns:
        pd.DataFrame: The normalized dataframe.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_pivot)
    return scaled_data, scaler


def create_sequences(data, seq_length):
    """Create sequences of data for training the RNN model.

    Args:
        data (np.ndarray): The data to create sequences from.
        seq_length (int): The sequence length.
    """
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i : i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        labels.append(label)

    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(
        labels, dtype=torch.float32
    )


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int,
) -> nn.Module:
    """Train the RNN model.

    Args:
        model (nn.Module): The RNN model.
        train_loader (DataLoader): The DataLoader for training data.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        epochs (int): The number of epochs to train the model.

    Returns:
        nn.Module: The trained RNN model.
    """
    for epoch in range(epochs):
        for seq, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(seq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


def predict(
    model: nn.Module,
    data: np.ndarray,
    seq_length: int,
    future_steps: int,
):
    """Predict future values using the trained RNN model.

    Args:
        model (nn.Module): The trained RNN model.
        data (np.ndarray): The data to predict future values for.
        seq_length (int): The sequence length.
        future_steps (int): The number of future steps to predict.
    """
    model.eval()
    predictions = []
    input_seq = torch.tensor(data[-seq_length:], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        for _ in range(future_steps):
            output = model(input_seq)
            predictions.append(output.squeeze().numpy())
            input_seq = torch.cat((input_seq[:, 1:, :], output.unsqueeze(0)), dim=1)
    return predictions


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


def train_rnn_model(
    df: pd.DataFrame,
    food_item: str,
    n_splits: int = 10,
    batch_size: int = 1,
    hidden_size: int = 6,
    learning_rate: float = 0.001,
    num_epochs: int = 100,
) -> Union[nn.Module, np.ndarray, MinMaxScaler]:
    """
    Processes a food item by normalizing data, creating sequences, splitting the data, and training a model.

    Args:
        df (pd.DataFrame): The dataframe containing the food item data.
        food_item (str): The food item to process.
        n_splits (int, optional): Number of splits for TimeSeriesSplit. Default is 10.
        batch_size (int, optional): Batch size for DataLoader. Default is 1.
        hidden_size (int, optional): Hidden size for the RNN model. Default is 6.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 0.001.
        num_epochs (int, optional): Number of epochs for training the model. Default is 100.

    Returns:
        nn.Module: The trained RNN model.
        np.ndarray: The normalized data.
        MinMaxScaler: The MinMaxScaler used for normalization.
    """
    food_data = load_data(df, food_item)
    data, scalar = normalize_data(food_data)
    sequences, labels = create_sequences(data, 3)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_index, test_index in tscv.split(sequences):
        train_sequences, test_sequences = sequences[train_index], sequences[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_sequences, train_labels),
            batch_size=batch_size,
            shuffle=False,
        )
        model = SimpleRNN(
            input_size=data.shape[1],
            hidden_size=hidden_size,
            output_size=data.shape[1],
        )
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_model(model, train_loader, criterion, optimizer, num_epochs)

        return model, data, scalar


def process_food_predictions(
    df: pd.DataFrame,
    future_predictions: np.ndarray,
    food_item: str,
) -> pd.DataFrame:
    """
    Processes future predictions for a specified food item by merging them with actual data,
    adjusting columns, and computing predicted prices.

    Args:
        df (pd.DataFrame): The dataframe containing the food item data.
        future_predictions (np.ndarray): Array of future predictions.
        food_item (str): The food item to process.

    Returns:
        pd.DataFrame: The dataframe with processed predictions and computed prices.
    """
    # Create DataFrame from predictions and adjust columns
    food_data = load_data(df, food_item)
    pred = pd.DataFrame(future_predictions, columns=food_data.columns)
    pred = pred.melt()
    pred["YEAR"] = 2022
    pred = pred.reset_index()
    pred = pred.rename(columns={"index": "MONTH", "value": "PRED_DIFF"})
    pred["MONTH"] = (pred.MONTH + 1) % 12
    pred["MONTH"] = np.where(pred["MONTH"] == 0, 12, pred["MONTH"])

    # Filter and merge data for the specified food item and year
    df_2022_food = df[(df.FOOD_ITEM == food_item) & (df.YEAR == 2022)].copy()
    df_2022_food = pd.merge(
        df_2022_food, pred, on=["MONTH", "YEAR", "COUNTRY"], how="left"
    )
    df_2022_food = df_2022_food.sort_values(by=["COUNTRY", "YEAR", "MONTH"])

    # Shift and calculate CPI differences
    df_2022_food["SHIFTED_CPI_DIFFERENCE"] = df_2022_food.groupby("COUNTRY")[
        "CPI_DIFFERENCE"
    ].shift(1)
    df_2022_food["PRED_CPI_DIFFERENCE"] = (
        df_2022_food["SHIFTED_CPI_DIFFERENCE"] + df_2022_food["PRED_DIFF"]
    )
    df_2022_food = df_2022_food.drop(columns=["SHIFTED_CPI_DIFFERENCE"])

    # Shift and calculate predicted CPI
    df_2022_food["SHIFTED_CPI"] = df_2022_food.groupby("COUNTRY")["CPI"].shift(1)
    df_2022_food["PRED_CPI"] = (
        df_2022_food["SHIFTED_CPI"] + df_2022_food["PRED_CPI_DIFFERENCE"]
    )
    df_2022_food = df_2022_food.drop(columns=["SHIFTED_CPI"])

    # Merge with original prices and calculate predicted prices
    original_prices = df[
        (df.FOOD_ITEM == food_item) & (df.YEAR == 2018) & (df.MONTH == 1)
    ].copy()
    original_prices = original_prices[
        ["COUNTRY", "YEAR", "FOOD_ITEM", "PRICE_IN_USD"]
    ].rename(columns={"PRICE_IN_USD": "ORIGINAL_PRICE"})
    df_2022_food = pd.merge(
        df_2022_food, original_prices, on=["COUNTRY", "FOOD_ITEM"], how="left"
    )
    df_2022_food["PRED_PRICE"] = (
        df_2022_food.PRED_CPI / 100
    ) * df_2022_food.ORIGINAL_PRICE

    return df_2022_food
