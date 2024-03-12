# Revised script to interpolate x_pos and y_pos columns and set 'event' to 'touchmove' for interpolated rows

import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# Function to downsample or interpolate the x_pos and y_pos to match exactly 128 rows per group
def process_group(group):
    n_points = 128
    # Determine whether to downsample or interpolate
    if len(group) > n_points:
        # Downsample the group to n_points
        indices = np.round(np.linspace(0, len(group) - 1, n_points)).astype(int)
        new_group = group.iloc[indices].reset_index(drop=True)
    elif len(group) < n_points:
        # Interpolate x_pos and y_pos to reach n_points
        f_x = interp1d(np.arange(len(group)), group['x_pos'], kind='linear')
        f_y = interp1d(np.arange(len(group)), group['y_pos'], kind='linear')
        new_x = f_x(np.linspace(0, len(group) - 1, n_points))
        new_y = f_y(np.linspace(0, len(group) - 1, n_points))
        
        # Create a new DataFrame with interpolated values
        new_group = pd.DataFrame(index=np.arange(n_points), columns=group.columns)
        new_group['x_pos'] = new_x
        new_group['y_pos'] = new_y
        new_group['event'] = 'touchmove'  # Set the 'event' column for interpolated rows to 'touchmove'
        
        # Fill other columns with the first row's values
        for col in group.columns.difference(['x_pos', 'y_pos', 'event']):
            new_group[col] = group[col].iloc[0]

        # Ensure the original rows retain their 'event' values
        original_indices = np.round(np.linspace(0, len(group) - 1, len(group))).astype(int)
        new_group.loc[original_indices, 'event'] = group['event'].values
    else:
        new_group = group

    return new_group

# Define the source and target directories
source_dir = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/datasets/datasets_per_word/test_datasets'  # Change this to your source directory path
target_dir = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/datasets/datasets_per_word/test_datasets2'  # Change this to your target directory path

# Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Process each CSV file in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith('.csv'):
        # Read the CSV file
        file_path = os.path.join(source_dir, filename)
        df = pd.read_csv(file_path)

        # Mark groups based on the 'sentence' column change
        df['temp_group'] = (df['sentence'] != df['sentence'].shift()).cumsum()

        # Process each group
        processed_df = df.groupby('temp_group').apply(process_group).reset_index(drop=True)

        # Drop the 'temp_group' column
        processed_df = processed_df.drop(columns='temp_group')

        # Save the processed DataFrame to the target directory
        target_file_path = os.path.join(target_dir, filename)
        processed_df.to_csv(target_file_path, index=False)

# Note: Replace the placeholder paths with the actual paths to your source and target directories.
