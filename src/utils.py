from dataclasses import dataclass
import os

import yaml
import pandas as pd
import matplotlib.colors as mcolors


def load_env(path: str = None) -> None:
    """Load environment variables from .env file.

    Assumes that the .env file is in the base directory of the project,
    otherwise the path to the .env file should be provided.

    Args:
        path (str): The path to the .env file.
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    with open(path, "r") as f:
        for line in f:
            # Skip comments and empty lines
            if line.startswith("#") or not line.strip():
                continue
            key, value = line.strip().split("=")
            os.environ[key] = value


def load_variables(path: str = None) -> None:
    """Load environment variables from variables.yml file.

    Assumes that the variables.yml file is in the base directory of the project,
    otherwise the path to the variables.yml file should be provided.

    Args:
        path (str): The path to the variables.yml file.
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "variables.yml")
    with open(path, "r") as f:
        variables = yaml.safe_load(f)
        for key, value in variables.items():
            os.environ[key] = value


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data by removing rows with missing values.

    In addition cleans column names to be all caps and fills
    spaces with underscores.

    Args:
        df (pd.DataFrame): The dataframe to clean.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    df = df.dropna()
    df.columns = df.columns.str.upper().str.replace(" ", "_")
    return df


@dataclass
class FoodPricesColumns:
    COUNTRY: str = "COUNTRY"
    FOOD_ITEM: str = "FOOD_ITEM"
    QUALITY: str = "QUALITY"
    CURRENCY: str = "CURRENCY"
    UNIT_OF_MEASURMENT: str = "UNIT_OF_MEASURMENT"
    YEAR: str = "YEAR"
    MONTH: str = "MONTH"
    AVERAGE_PRICE: str = "AVERAGE_PRICE"
    PRICE_IN_USD: str = "PRICE_IN_USD"
    AVAILABILITY: str = "AVAILABILITY"

    @property
    def columns(self: object) -> list[str]:
        return [
            self.COUNTRY,
            self.FOOD_ITEM,
            self.QUALITY,
            self.CURRENCY,
            self.UNIT_OF_MEASURMENT,
            self.YEAR,
            self.MONTH,
            self.PRICE,
            self.AVAILABILITY,
        ]

    @property
    def categorical_columns(self: object) -> list[str]:
        return [
            self.COUNTRY,
            self.FOOD_ITEM,
            self.QUALITY,
            self.CURRENCY,
            self.UNIT_OF_MEASURMENT,
        ]

    @property
    def numerical_columns(self: object) -> list[str]:
        return [self.YEAR, self.MONTH, self.PRICE]


cols = FoodPricesColumns()


@dataclass
class Theme:
    TRANSPARENT: str = "rgba(0, 0, 0, 0)"

    TEAL: str = "#60C090"
    TEAL_LIGHTER_80: str = "#DFF2E9"
    TEAL_LIGHTER_60: str = "#BFE6D3"
    TEAL_LIGHTER_40: str = "#A0D9BC"
    TEAL_DARKER_25: str = "#3D9B6C"
    TEAL_DARKER_50: str = "#296748"

    DARK_TEAL: str = "#215649"
    DARK_TEAL_LIGHTER_80: str = "#C6E9E1"
    DARK_TEAL_LIGHTER_60: str = "#8ED4C3"
    DARK_TEAL_LIGHTER_40: str = "#55BEA4"
    DARK_TEAL_DARKER_25: str = "#194037"
    DARK_TEAL_DARKER_50: str = "#102B25"

    GREEN: str = "#97DF65"
    GREEN_LIGHTER_80: str = "#EAF9E0"
    GREEN_LIGHTER_60: str = "#D5F2C1"
    GREEN_LIGHTER_40: str = "#C1ECA3"
    GREEN_DARKER_25: str = "#6BC92A"
    GREEN_DARKER_50: str = "#47861C"

    YELLOW: str = "#FFFF00"
    YELLOW_LIGHTER_80: str = "#FFFFCC"
    YELLOW_LIGHTER_60: str = "#FFFF99"
    YELLOW_LIGHTER_40: str = "#FFFF66"
    YELLOW_DARKER_25: str = "#BFBF00"
    YELLOW_DARKER_50: str = "#7F7F0"

    TURQUOISE: str = "#00B0F0"
    TURQUOISE_LIGHTER_80: str = "#C9F1FF"
    TURQUOISE_LIGHTER_60: str = "#93E2FF"
    TURQUOISE_LIGHTER_40: str = "#5DD4FF"
    TURQUOISE_DARKER_25: str = "#0084B4"
    TURQUOISE_DARKER_50: str = "#005878"

    ACCENT_GRAY: str = "#797979"
    ACCENT_GRAY_LIGHTER_80: str = "#E4E4E4"
    ACCENT_GRAY_LIGHTER_60: str = "#C9C9C9"
    ACCENT_GRAY_LIGHTER_40: str = "#AFAFAF"
    ACCENT_GRAY_DARKER_25: str = "#5B5B5B"
    ACCENT_GRAY_DARKER_50: str = "#3C3C3C"

    BACKGROUND_WHITE: str = "#FFFFFF"
    BACKGROUND_WHITE_DARKER_5: str = "#F2F2F2"
    BACKGROUND_WHITE_DARKER_15: str = "#D9D9D9"
    BACKGROUND_WHITE_DARKER_25: str = "#BFBFBF"
    BACKGROUND_WHITE_DARKER_35: str = "#A6A6A6"
    BACKGROUND_WHITE_DARKER_50: str = "#7F7F7F"

    DARK_GRAY: str = "#575757"
    DARK_GRAY_LIGHTER_80: str = "#DDDDDD"
    DARK_GRAY_LIGHTER_60: str = "#BCBCBC"
    DARK_GRAY_LIGHTER_40: str = "#9A9A9A"
    DARK_GRAY_DARKER_25: str = "#414141"
    DARK_GRAY_DARKER_50: str = "#2C2C2C"

    FONT_PRIMARY: str = "Helvetica"
    FONT_SECONDARY: str = "Playfair Display"

    @staticmethod
    def generate_colorscale(start: str, end: str, len: int = 255) -> list[str]:
        """Creates a colorscale gradient between two colors.

        Args:
            start (str): The starting color in hex format.
            end (str): The ending color in hex format.
            len (int): The number of colors to generate in the gradient.

        Returns:
            list[str]: A list of colors in hex format for the
                gradient between the two colors.
        """
        start_rgb = mcolors.hex2color(start)
        end_rgb = mcolors.hex2color(end)
        colors = [
            mcolors.rgb2hex(
                [
                    start_rgb[j] + (float(i) / (len - 1)) * (end_rgb[j] - start_rgb[j])
                    for j in range(3)
                ]
            )
            for i in range(len)
        ]
        return colors


theme = Theme()
