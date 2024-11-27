import pandas as pd

# Load the CSV file
file_path = '30-graph_gurus_evaluation/Cluster CV/metrics.csv'
#file_path = '5-derm.ai_evaluation/Cluster CV/metrics.csv'
data = pd.read_csv(file_path)

# Calculate mean and standard deviation for each metric
metrics = data.columns[1:]  # Exclude the 'ID' column
results = {}

for metric in metrics:
    mean_value = data[metric].mean()
    std_dev_value = data[metric].std()
    results[metric] = {'mean': mean_value, 'std_dev': std_dev_value}

# Prepare the results for saving
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Metric'}, inplace=True)

# Save the results to a new CSV file
output_file_path = file_path[:-4] + '_summary.csv'
results_df.to_csv(output_file_path, index=False)

output_file_path
