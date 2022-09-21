import pathlib
import pandas as pd
import dataframe_image as dfi
import numpy as np
import matplotlib.pyplot as plt


def missing_values_in_data_frame(df):
    missing_values = df.isnull().sum()

    missing_values_percent = 100 * df.isnull().sum() / len(df)

    missing_values_table = pd.concat([missing_values, missing_values_percent], axis=1)

    missing_values_table_rennamed_columns = missing_values_table.rename(
        columns={0: "Missing values", 1: "Percent of total values"}
    )

    missing_values_table_rennamed_columns = (
        missing_values_table_rennamed_columns[
            missing_values_table_rennamed_columns.iloc[:, 1] != 0
        ]
        .sort_values("Percent of total values", ascending=False)
        .round(1)
    )

    print(
        "Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
        "There are "
        + str(missing_values_table_rennamed_columns.shape[0])
        + " columns that have missing values."
    )

    return missing_values_table_rennamed_columns


dataFrame = pd.read_csv("{}/data/data.csv".format(pathlib.Path(__file__).parents[0]))

# To delete null values and their columns
dataFrame = dataFrame.replace({"Not Available": np.nan})
for col in list(dataFrame.columns):
    if (
        "ft²" in col
        or "kBtu" in col
        or "therms" in col
        or "kWh" in col
        or "gal" in col
        or "Metric Tons CO2e" in col
        or "Score" in col
    ):
        dataFrame[col] = dataFrame[col].astype(float)
missing_data_frame = missing_values_in_data_frame(dataFrame)
missing_columns = list(
    missing_data_frame[missing_data_frame["Percent of total values"] > 50].index
)
dataFrame = dataFrame.drop(columns=list(missing_columns))

# To delete extremals or experemental inputs
third_quartile = dataFrame["Site EUI (kBtu/ft²)"].describe()["75%"]
first_quartile = dataFrame["Site EUI (kBtu/ft²)"].describe()["25%"]
iqr = third_quartile - first_quartile
data = dataFrame[
    (dataFrame["Site EUI (kBtu/ft²)"] > (first_quartile - 3 * iqr))
    & (dataFrame["Site EUI (kBtu/ft²)"] < (third_quartile + 3 * iqr))
]

# To rename column
dataFrame = dataFrame.rename(columns={"ENERGY STAR Score": "score"})

# Show data
# dfi.export(data, "data.png", max_rows=100, max_cols=30)
dataFrame.info()
#####################
