from src.data_loader import load_and_clean_data

df = load_and_clean_data()
print(df.head())
print(f"Тип TotalCharges: {df['TotalCharges'].dtype}")