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
    def __init__(self, input_dim):
        super(WeightedRankNet, self).__init__()

        # Define the weights as fixed parameters
        self.bm25_k1 = nn.Parameter(torch.tensor(3.5, dtype=torch.float32), requires_grad=True)
        self.bm25_b = nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)
        self.bm25_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
        
        self.page_rank = nn.Parameter(torch.tensor(0.2, dtype=torch.float32), requires_grad=True)
        self.in_link = nn.Parameter(torch.tensor(0.1, dtype=torch.float32), requires_grad=True)
        self.out_link = nn.Parameter(torch.tensor(0.1, dtype=torch.float32), requires_grad=True)

        self.freshness = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)

        # Define the model layers
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output relevance score
        )

    def forward(self, batch_indices):
        """
        Compute scores for the documents in the current batch based on their indices (doc IDs).
        """
        global global_features  # Use global features
        avg_doc_len, total_docs = fetch_total_doc_statistics_mslr().values()  # Corpus stats
        total_docs_tensor = torch.tensor(float(total_docs), dtype=torch.float32)  # Convert to tensor
        avg_doc_len_tensor = torch.tensor(float(avg_doc_len), dtype=torch.float32)  # Convert to tensor
        weights = {
            "bm25_params": {"k1": self.bm25_k1, "b": self.bm25_b},
            "bm25": self.bm25_weight,
            "pageRank": {
                "pageRank": self.page_rank,
                "inLink": self.in_link,
                "outLink": self.out_link,
            },
            "metadata": {"freshness": self.freshness},
        }

        # Ensure batch_indices is a list of integers
        if isinstance(batch_indices, torch.Tensor):
            batch_indices = batch_indices.tolist()  # Convert tensor to list

        scores = []
        for doc_id in batch_indices:  # Process each doc_id in the batch
            feature_vector = global_features[doc_id]  # Access feature vector directly

            # Extract query and document information
            query = str(feature_vector[4])  # Feature 5 is the query term
            metadata = fetch_doc_metadata_mslr(doc_id)["metadata"]
            pagerank_data = fetch_pagerank_mslr(metadata["URL"])

            # BM25 scoring
            term_freq = torch.tensor(float(feature_vector[24]), dtype=torch.float32)  # Term frequency
            doc_length = torch.tensor(float(metadata["docLength"]), dtype=torch.float32)  # Document length
            num_docs_with_term = torch.tensor(len(global_features), dtype=torch.float32)  # Total number of docs

            idf_term = torch.log((total_docs_tensor - num_docs_with_term + 0.5) / (num_docs_with_term + 0.5) + 1)
            numerator = term_freq * (weights["bm25_params"]["k1"] + 1)
            denominator = term_freq + weights["bm25_params"]["k1"] * (
                1 - weights["bm25_params"]["b"] + weights["bm25_params"]["b"] * (doc_length / avg_doc_len_tensor)
            )
            bm25_score = idf_term * (numerator / denominator)

            # Pagerank scoring
            pagerank_score = (
                weights["pageRank"]["pageRank"] * torch.tensor(float(pagerank_data.get("pageRank", 0.0)), dtype=torch.float32) +
                weights["pageRank"]["inLink"] * torch.tensor(float(pagerank_data.get("inLinkCount", 0.0)), dtype=torch.float32) +
                weights["pageRank"]["outLink"] * torch.tensor(float(pagerank_data.get("outLinkCount", 0.0)), dtype=torch.float32)
            )

            # Metadata scoring (Simple Freshness Placeholder)
            freshness_score = torch.rand(1).item() * weights["metadata"]["freshness"]

            # Combine scores
            combined_score = weights["bm25"] * bm25_score + pagerank_score + freshness_score
            scores.append(combined_score)

        # Convert scores to a tensor
        scores_tensor = torch.stack(scores).unsqueeze(1)  # Stack to create a single tensor
        return scores_tensor
    
    def get_weights(self):
        return {
            "bm25_params": {"k1": self.bm25_k1.item(), "b": self.bm25_b.item()},
            "bm25": self.bm25_weight.item(),
            "pageRank": {
                "pageRank": self.page_rank.item(),
                "inLink": self.in_link.item(),
                "outLink": self.out_link.item(),
            },
            "metadata": {"freshness": self.freshness.item()},
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

def train_model(features, labels, epochs=5, batch_size=64, lr=0.001):
    """
    Train the WeightedRankNet model using rank_documents directly in the forward pass.
    """
    # Create a dataset with document indices and labels
    model = WeightedRankNet(input_dim=features.shape[1])
    dataset = TensorDataset(torch.arange(len(global_features)), torch.tensor(labels, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.L1Loss()  # Mean Absolute Error Loss

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_indices, batch_labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            optimizer.zero_grad()

            # Compute predictions for the current batch
            predictions = model(batch_indices)  # Pass document indices to the model

            # Compute loss
            loss = loss_fn(predictions, batch_labels.unsqueeze(1))  # Match dimensions for L1Loss
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        # Epoch metrics
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
    train = load_data('../data/Fold1/train.txt', 100000)
    feats, labels = extract_features_and_labels(train)

    # Train the model
    global_features = feats
    model = train_model(feats, labels, epochs=40, batch_size=128, lr=0.00001)

    # Print optimized weights
    optimized_weights = model.get_weights()
    print("Optimized Weights:", optimized_weights)
