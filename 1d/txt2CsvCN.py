import pandas as pd
import numpy as np
import sys
# Specify the file path
file_path = 'CNDiffData.txt'  # Replace with the correct path if needed

# Load the data into a DataFrame
data = pd.read_csv(file_path, delim_whitespace=True, header=None)

# Extract row indices, column indices, and data
row_indices = data[0].astype(int)
col_indices = data[1].astype(int)
values = data[3]  # The fourth column contains the data you want in the CSV

# Determine the size of the matrix
max_row = row_indices.max()
max_col = col_indices.max()

# Create an empty matrix filled with NaNs
matrix = np.full((max_row + 1, max_col + 1), np.nan)

# Fill the matrix with the data
for row, col, value in zip(row_indices, col_indices, values):
    matrix[int(row), int(col)] = value

# Convert the matrix to a DataFrame
df = pd.DataFrame(matrix)

# Save the DataFrame to a CSV file
output_csv_path = 'CNDiffData_matrix.csv'  # Replace with the desired output path
df.to_csv(output_csv_path, index=False, header=False)

print(f"Data has been saved to {output_csv_path}")
