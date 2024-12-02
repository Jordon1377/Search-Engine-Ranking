# Dataset: https://www.microsoft.com/en-us/research/project/mslr/
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

COLUMN_NAMES = ['label', 'qid'] + [f'feature{i}' for i in range(1, 137)]

def load_data(file_relative_path, num_rows=None):
    """ Loads the dataset from a relative path """
    data_file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), file_relative_path)
    )
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found at {data_file_path}")
    
    # Define column names
    column_names = ['label', 'qid'] + [f'feature{i}' for i in range(1, 137)]
    
    # Load the dataset
    data = pd.read_csv(
        data_file_path,
        sep='\s+',
        header=None,
        names=COLUMN_NAMES,
        engine='c',
        nrows=num_rows
    )
    return data
    
def plt_class_dist(data):
    # Checking the distribution of labels
    label_counts = train['label'].value_counts()
    
    # Plotting class distribution
    plt.figure(figsize=(8, 6))
    label_counts.sort_index().plot(kind='bar')
    plt.title('Class Distribution of Relevance Labels')
    plt.xlabel('Relevance Label')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()

def plt_feture_dist(data):
    # Checking the distribution of features
    plt.figure(figsize=(8, 6))
    train['qid'].value_counts().plot(kind='bar')
    plt.title('Number of Documents per Query ID (Top 20 QIDs)')
    plt.xlabel('Query ID')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    # Load the first 1000 data points from the data set
    train = load_data('../data/Fold1/train.txt', 1000)

    # Print the first 5 rows
    print(train.head())

    # Extract numeric values from features and calculate descriptive statistics
    feature_stats = train.iloc[:, 2:].apply(
        lambda col: pd.to_numeric(col.str.split(':').str[1], errors='coerce')
    ).describe()

    # Print the summary
    print("\nStatistical Summary of Features:")
    print(feature_stats)

    plt_class_dist(train)
    plt_feture_dist(train)
