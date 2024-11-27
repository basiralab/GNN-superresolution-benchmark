###
#!Note: This script is used to combine the results of 3 fold Cluster-CVs and Random-CVs
# and calculate the mean and standard deviation of the results.
# Change the file names to the ones you want to combine.
###

import pandas as pd

# change to 3-fold output file path.
csv1 = pd.read_csv('ID-randomCluster2-0-fixed.csv')
csv2 = pd.read_csv('ID-randomCluster2-1-fixed.csv')
csv3 = pd.read_csv('ID-randomCluster2-0-fixed.csv')

csv1.set_index('ID', inplace=True)
csv2.set_index('ID', inplace=True)
csv3.set_index('ID', inplace=True)

combined_df = pd.concat([csv1, csv2, csv3])

mean_values = combined_df.mean()
std_dev_values = combined_df.std()

print("Average:\n", mean_values)
print("Standard Deviation:\n", std_dev_values)


combined_df.to_csv('combined_data_CLUSTER.csv')


