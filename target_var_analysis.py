import pandas as pd

# Load your dataset
df = pd.read_csv('KSA Real Estate - Dataset.csv')

# Assuming 'Price' column values are stored as strings with commas,
# Convert them to numeric after removing commas
df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')

# Calculate the required statistics
average_price = df['Price'].mean()
median_price = df['Price'].median()
std_dev_price = df['Price'].std()
variance_price = df['Price'].var()

# Print the statistics
print(f"Average Price: {average_price}")
print(f"Median Price: {median_price}")
print(f"Standard Deviation: {std_dev_price}")
print(f"Variance: {variance_price}")
