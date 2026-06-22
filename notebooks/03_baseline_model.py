import pandas as pd

df = pd.read_excel(
    "data/cleaned_flight_summary.xlsx"
)

print(df.head())

print(df.shape)