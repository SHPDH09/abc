import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.preprocessing import REQUIRED_COLUMNS

def plot_burnout_distribution(df):
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='BurnoutRisk', palette='coolwarm', ax=ax)
    ax.set_title("Burnout Risk Count")
    return fig

def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    corr = df[REQUIRED_COLUMNS].corr()
    sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    return fig
