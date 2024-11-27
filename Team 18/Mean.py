import pandas as pd

# Load the CSV file
file_path = 'ID_randomCV.csv'
df = pd.read_csv(file_path)

# Calculate mean and standard deviation for each column, excluding the first column
mean_values = df.iloc[:, 1:].mean()
std_values = df.iloc[:, 1:].std()

# Create DataFrames for mean and standard deviation
mean_row = pd.DataFrame([['mean'] + mean_values.tolist()], columns=df.columns)
std_row = pd.DataFrame([['std'] + std_values.tolist()], columns=df.columns)

# Append the new rows to the dataframe
result_df = pd.concat([df, mean_row, std_row], ignore_index=True)

# Save the resulting dataframe to a new CSV file
output_file_path = 'ID_randomCV.csv'
result_df.to_csv(output_file_path, index=False)

print(f"The new CSV file has been saved as {output_file_path}")


