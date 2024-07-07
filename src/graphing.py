import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from utils import theme


def _inter_country_price_heatmap_annotation_helper(value: float, statistic: str) -> str:
    """Helper function to format the annotations for the heatmaps.

    Args:
        value (float): The value to format.
        statistic (str): The statistic to format.

    Returns:
        str: The formatted value.
    """
    match statistic:
        case "AVERAGE_PRICE":
            return f"${value:.2f}"
        case "PERC_PRICE_CHANGE_START_YEAR_TO_END_YEAR":
            return f"{value:.2f}%"
        case "MAX_INTER_MONTH_PRICE_CHANGE":
            return f"${value:.2f}"
        case "MAX_OVERALL_PRICE_CHANGE":
            return f"${value:.2f}"
        case "PRICE_STANDARD_DEVIATION":
            return f"{value:.2f}"
        case "PRICE_VARIANCE":
            return f"{value:.2f}"
        case _:
            raise ValueError(f"Unsupported statistic: {statistic}")


def _inter_country_price_heatmap_title_helper(statistic: str) -> str:
    """Helper function to format the title for the heatmaps.

    Args:
        statistic (str): The statistic to format.

    Returns:
        str: The formatted value.
    """
    match statistic:
        case "AVERAGE_PRICE":
            return "Average Prices"
        case "PERC_PRICE_CHANGE_START_YEAR_TO_END_YEAR":
            return "Percentage Price Change"
        case "MAX_INTER_MONTH_PRICE_CHANGE":
            return "Max Inter-Month Price Change"
        case "MAX_OVERALL_PRICE_CHANGE":
            return "Max Overall Price Change"
        case "PRICE_STANDARD_DEVIATION":
            return "Price Standard Deviation"
        case "PRICE_VARIANCE":
            return "Price Variance"
        case _:
            raise ValueError(f"Unsupported statistic: {statistic}")


def inter_country_price_heatmaps(
    prices: pd.DataFrame,
    normalized_prices: pd.DataFrame,
    food_item: str,
    statistic: str,
) -> None:
    """Create two heatmaps to compare prices and normalized prices.

    The first heatmap shows the average prices for each country and year. The second
    heatmaps is normalized by the average price for that good across all countries for
    that particular year.

    Args:
        prices (pd.DataFrame): The prices for each country and year.
        normalized_prices (pd.DataFrame): The normalized prices for each country and

    Returns:
        None
    """
    # Extract the data for first heatmap
    x_axis1 = prices.columns.astype(str)  # years
    y_axis1 = prices.index  # countries
    z_values1 = prices.values  # prices

    plot_title = _inter_country_price_heatmap_title_helper(statistic)

    # Create annotations for the first heatmap
    annotations1 = []
    for i, row in enumerate(z_values1):
        for j, value in enumerate(row):
            annotations1.append(
                go.layout.Annotation(
                    text=_inter_country_price_heatmap_annotation_helper(
                        value,
                        statistic,
                    ),
                    x=x_axis1[j],
                    y=y_axis1[i],
                    xref="x1",
                    yref="y1",
                    showarrow=False,
                    font=dict(color="black"),
                )
            )

    x_axis2 = normalized_prices.columns.astype(str)  # years
    y_axis2 = normalized_prices.index  # countries
    z_values2 = normalized_prices.values  # prices

    # Create annotations for the second heatmap
    annotations2 = []
    for i, row in enumerate(z_values2):
        for j, value in enumerate(row):
            annotations2.append(
                go.layout.Annotation(
                    text=f"{value:.2f}",
                    x=x_axis2[j],
                    y=y_axis2[i],
                    xref="x2",
                    yref="y2",
                    showarrow=False,
                    font=dict(color="black"),
                )
            )

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"{plot_title} Heatmap", f"Normalized {plot_title} Heatmap"),
    )

    # Create the first heatmap
    fig.add_trace(
        go.Heatmap(
            z=z_values1,
            x=x_axis1,
            y=y_axis1,
            colorscale=theme.generate_colorscale(
                theme.BACKGROUND_WHITE_DARKER_5, theme.TEAL_DARKER_25
            ),
            showscale=False,
            hovertemplate="Country: %{y}<br>Year: %{x}<br>Value: $%{z:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Create the second heatmap
    fig.add_trace(
        go.Heatmap(
            z=z_values2,
            x=x_axis2,
            y=y_axis2,
            colorscale="Blues",
            showscale=False,
            hovertemplate="Country: %{y}<br>Year: %{x}<br>Value: %{z:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Adjust subplot title positions
    for annotation in fig["layout"]["annotations"]:
        if "Heatmap" in annotation["text"]:  # Check for the titles
            annotation[
                "y"
            ] += 0.075  # Move the title further up (adjust value as needed)

    # Update layout to ensure each box is square and add spacing
    fig.update_layout(
        title_text=f"Heatmaps of {plot_title} for {food_item} by Country and Year",
        annotations=annotations1 + annotations2,
        width=1100,
        height=500,
        margin=dict(l=100, r=100, t=100, b=100),  # Adjust the top margin if needed
    )

    fig.update_xaxes(
        dict(
            side="top",
            tickmode="array",
            tickvals=list(range(len(x_axis1))),
            ticktext=x_axis1,
            scaleanchor="y",
            scaleratio=1,
            title_font=dict(size=14, family="Arial"),
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(
        dict(
            side="top",
            tickmode="array",
            tickvals=list(range(len(x_axis2))),
            ticktext=x_axis2,
            scaleanchor="y",
            scaleratio=1,
            title_font=dict(size=14, family="Arial"),
        ),
        row=1,
        col=2,
    )

    fig.update_yaxes(
        dict(side="left", ticks="", tickfont=dict(family="sans-serif"), ticksuffix=" "),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        dict(side="left", ticks="", tickfont=dict(family="sans-serif"), ticksuffix=" "),
        row=1,
        col=2,
    )

    fig.show()
