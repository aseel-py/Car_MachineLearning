test_df = pd.read_csv("test_cleaned.csv")

for col in test_df.select_dtypes(include="object").columns:
    test_df[col] = LabelEncoder().fit_transform(test_df[col])

test_df = test_df[X.columns]

predicted_prices = model.predict(test_df)

print(predicted_prices[:5])
