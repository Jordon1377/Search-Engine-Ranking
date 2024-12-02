import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
import numpy as np

from eda import load_data
from rank import rank_documents

# Model -----------------------------------------------------------------------
class WeightedRankNet(nn.Module):
    def __init__(self, input_dim, weights):
        super(WeightedRankNet, self).__init__()
        self.weights = nn.ParameterDict({
            key: (
                nn.Parameter(torch.tensor(value, dtype=torch.float32))
                if isinstance(value, float)
                else nn.ParameterDict({
                    k: nn.Parameter(torch.tensor(v, dtype=torch.float32))
                    for k, v in value.items()
                })
                if isinstance(value, dict)
                else None
            )
            for key, value in weights.items()
        })
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output relevance score
        )

    def forward(self, x):
        return self.model(x)
    
    def get_weights(self):
        return {
            key: {k: v.item() for k, v in value.items()}
            if isinstance(value, nn.ParameterDict)
            else value.item()
            for key, value in self.weights.items()
        }

# Features to API (From List on website)---------------------------------------
def fetch_pagerank_mslr(doc_id):
    """
    Simulate fetch_pagerank API using MSLR features.
    """
    # Ensure doc_id is within bounds
    if doc_id < 0 or doc_id >= len(global_features):
        return {"pageRank": 0.0, "inLinkCount": 0, "outLinkCount": 0}

    # Use the feature vector to extract the PageRank-related data
    feature_vector = global_features[doc_id]
    return {
        "pageRank": feature_vector[129],  # Feature 130
        "inLinkCount": feature_vector[127],  # Feature 128
        "outLinkCount": feature_vector[128]  # Feature 129
    }

# Cache to store precomputed statistics
cached_statistics = None
def fetch_total_doc_statistics_mslr():
    """
    Fetch global document statistics.
    This function computes and caches the statistics based on the features.
    """
    global cached_statistics
    
    # Return cached results if available
    if cached_statistics is not None:
        return cached_statistics

    if global_features is None:
        return {"avgDocLength": 0.0, "docCount": 0}

    whole_doc_len_index = 16
    avg_doc_length = global_features[:, whole_doc_len_index].mean()
    doc_count = len(global_features)

    # Cache the results
    cached_statistics = {"avgDocLength": avg_doc_length, "docCount": doc_count}

    return cached_statistics

def fetch_doc_metadata_mslr(doc_id):
    """
    Fetch metadata for a document given its docID (row number).
    Generates docLength from global features and constructs URL dynamically.
    """
    if doc_id < 0 or doc_id >= len(global_features):
        return None

    # Use the feature vector for this doc_id
    feature_vector = global_features[doc_id]
    
    # Feature index for "whole document length" (feature 15)
    whole_doc_index = 14

    metadata = {
        "docID": str(doc_id),
        "metadata": {
            "docLength": feature_vector[whole_doc_index],
            "URL": doc_id, # Page rank wants a url
        }
    }

    return metadata

def fetch_relevant_docs_mslr(term):
    """
    Fetch relevant documents for a given term.
    Constructs the invertible index for all docs containing the term.
    """
    if global_features is None:
        return None

    # Feature indices for covered query terms and term frequency
    covered_term_index = 4  # Feature 5
    frequency_index = 24    # Feature 25

    # Check if the term corresponds to a valid feature
    try:
        term_value = int(term)  # Convert term to an integer for matching
    except ValueError:
        return None

    # Build the index for the given term
    index = []
    for doc_id, feature_vector in enumerate(global_features):
        if feature_vector[covered_term_index] == term_value:
            index.append({
                "docID": str(doc_id),
                "frequency": feature_vector[frequency_index],
                "positions": []  # Positions are not used
            })

    if not index:
        return None

    return {"term": term, "index": index}

# Train Function --------------------------------------------------------------
def preprocess_data(features, labels):
    """ Converts features and labels into a TensorDataset for PyTorch. """
    # Convert features and labels to PyTorch tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # Create a TensorDataset
    return TensorDataset(features_tensor, labels_tensor)

def train_model(feats, labels, initial_weights, epochs=5, batch_size=64, lr=0.001):
    """
    Train the WeightedRankNet model using MSLR-based fetch functions for ranking.
    """
    dataset = preprocess_data(feats, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = WeightedRankNet(input_dim=feats.shape[1], weights=initial_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_features, batch_labels in tqdm(dataloader, desc="Training", leave=False):
            optimizer.zero_grad()

            # Predict relevance scores
            predictions = model(batch_features)

            # Retrieve dynamically ranked documents using current weights
            current_weights = model.get_weights()
            ranked_results = rank_documents(
                query=str(global_features[4]), # Feature 5
                weights=current_weights,
                fetch_total_doc_statistics=fetch_total_doc_statistics_mslr,
                fetch_relevant_docs=fetch_relevant_docs_mslr,
                fetch_doc_metadata=fetch_doc_metadata_mslr,
                fetch_pagerank=fetch_pagerank_mslr,
            )

            # Compute loss based on predictions and labels
            loss = loss_fn(predictions, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}")

    return model

# Main ------------------------------------------------------------------------
def extract_features_and_labels(train):
    """ Extracts features and labels from the dataset. """
    # Extract labels
    labels = train['label'].values

    # Extract features: parse the "index:value" format and convert to a dense matrix
    features = train.iloc[:, 2:].apply(
        lambda col: pd.to_numeric(col.str.split(':').str[1], errors='coerce')
    ).values

    return features, labels

global_features = None

if __name__ == "__main__":
    # Initial values
    initial_weights = {
        "bm25_params": {
            "k1": 3.5,
            "b": 0.5},
        "bm25": 1.0,
        "pageRank": {
            "pageRank": 0.2,
            "inLink": 0.1,
            "outLink": 0.1
        },
        "metadata": {
            "freshness": 1.0,
        }
    }

    #TODO: Update this for actual testing
    train = load_data('../data/Fold1/train.txt', 1000)
    feats, labels = extract_features_and_labels(train)

    # Train the model
    global_features = feats
    model = train_model(feats, labels, initial_weights, epochs=20)

    # Print optimized weights
    optimized_weights = model.get_weights()
    print("Optimized Weights:", optimized_weights)
