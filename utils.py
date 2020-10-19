import numpy as np
import pandas as pd

def basic_data_check(df):
    """Tells shape, info, describe and null values

    Args:
        df (pandas.DataFrame): Input dataframe for which info is needed
    """
    # Shape
    print("Shape")
    print(df.shape)
    print("-"*35)

    # Info
    print("Info")
    print(df.info())
    print("-"*35)

    # describe
    print("Describe")
    print(df.describe())
    print("-"*35)

    # null values
    print("Null Values in percentage")
    print(df.isnull().sum()/len(df)**100)
    print("-"*35)

def data_clean(df):
    """Cleaning data, by dropping columns and reducing cardinality.


    Args:
        df (pandas.DataFrame): Input, unclean data

    Returns:
        pandas.DataFrame: clean data
    """
    # dropping columns

    df.dropna(inplace=True)

    drop_columns = ["Daily Summary", "Pressure (millibars)", "Loud Cover"]
    df.drop(columns=drop_columns, axis=1, inplace=True)

    # Making new columns from Formatted Data
    # year 
    df["year"] = df["Formatted Date"].map(lambda x: x.split("-")[0])

    # month
    df["month"] = df["Formatted Date"].map(lambda x: x.split("-")[1])

    # day of month
    df["day"] = df["Formatted Date"].map(lambda x: x.split("-")[2].split()[0])

    # removing Formatted Date
    df.drop(columns=["Formatted Date"], axis=1, inplace=True)     

    # Reducing cardinality from the features
    top_summaries = df["Summary"].value_counts()[:5]
    df["Summary"] = df["Summary"].map(lambda x: x if x in top_summaries else "other")

    df["Summary"].value_counts()
    
    return df

def get_X_y(df):
    """Getting independent and dependent feature

    Args:
        df (pandas.DataFrame): DataFrame

    Returns:
        pandas.DataFrame/series: Independent and Dependent features
    """
    X = df.drop("Precip Type", axis=1)
    y = df["Precip Type"]   
    return X, y 

