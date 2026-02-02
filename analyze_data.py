import pandas as pd
try:
    df = pd.read_csv("my_system_data.csv")
    print("Data Loaded. Stats by Label:")
    print(df.groupby('label')['ram'].describe())
    print("\nMean CPU by Label:")
    print(df.groupby('label')['cpu'].describe())
except Exception as e:
    print(e)
