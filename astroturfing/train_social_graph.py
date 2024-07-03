import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import os
import utils
import json
import shutil


class GraphSAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.linear = torch.nn.Linear(out_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        node_embeddings = x
        x = self.linear(x)
        return torch.sigmoid(x), node_embeddings

def create_data(vertices_df, edges_df, labels, target_encoding):
    edge_index = torch.tensor(edges_df[['source', 'target']].values.T, dtype=torch.long)
    x = torch.tensor(vertices_df.values, dtype=torch.float)
    y = torch.tensor(target_encoding.transform(labels).astype(np.float32), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def infer(model, data_loader, device='cpu'):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            _, node_embeddings = model(data.x, data.edge_index)
            embeddings.append(node_embeddings.cpu().numpy())
    return np.vstack(embeddings)

def reindex_edges(vertices_df, edges_df):
    """Reindex the edges to the correct range, starting from zero.

    This function ensures that all node indices in the edges dataframe are within
    the valid range by mapping the original node indices to a new range starting from zero.
    It also verifies that all edges refer to valid nodes present in the vertices dataframe.

    Parameters:
    vertices_df (pd.DataFrame): DataFrame containing node features, indexed by node IDs.
    edges_df (pd.DataFrame): DataFrame containing edges with 'source' and 'target' columns.

    Returns:
    vertices_df (pd.DataFrame): Reindexed vertices DataFrame with continuous node indices.
    edges_df (pd.DataFrame): Reindexed edges DataFrame with updated 'source' and 'target' columns.
    
    Raises:
    ValueError: If there are any edges containing indices not found in the vertices DataFrame.
    """
    # Create a mapping from original node indices to new indices starting from zero
    node_mapping = {original_index: new_index for new_index, original_index in enumerate(vertices_df.index)}
    inverse_node_mapping = {new_index: original_index for new_index, original_index in enumerate(vertices_df.index)}

    # Apply the mapping to the edges dataframe
    edges_df['source'] = edges_df['source'].map(node_mapping)
    edges_df['target'] = edges_df['target'].map(node_mapping)

    # Check if there are any unmapped values (this should not happen if all indices are correct)
    if edges_df['source'].isnull().any() or edges_df['target'].isnull().any():
        raise ValueError("Some edges contain indices not found in the vertices dataframe")

    return vertices_df.reset_index(drop=True), edges_df, inverse_node_mapping

def plot_tsne_node_embeddings(model, loader, device, epoch):
    model.eval()
    node_embeddings = []
    labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            _, node_embeds = model(data.x, data.edge_index)
            node_embeddings.append(node_embeds.cpu().numpy())
            labels.append(data.y.cpu().numpy())

    node_embeddings = np.concatenate(node_embeddings, axis=0)
    labels = np.concatenate(labels, axis=0).flatten()  # Ensure labels are 1D
    tsne = TSNE(n_components=2, perplexity=5, init="pca", learning_rate="auto", random_state=42)
    embeddings_2d = tsne.fit_transform(node_embeddings)
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings_2d[labels == 0, 0], embeddings_2d[labels == 0, 1], label='Non-astroturfers', alpha=0.5)
    plt.scatter(embeddings_2d[labels == 1, 0], embeddings_2d[labels == 1, 1], label='Astroturfers', alpha=0.5)
    plt.legend()
    plt.title(f't-SNE of Node Embeddings at Epoch {epoch}')
    plt.savefig(f'tsne_social_graph/tsne_node_epoch_{epoch}.png')
    plt.close()

def run(args):
    user_labels_dir = 'produced_data/user_labels'

    train_edges_df = utils.read_pickle_from_file(os.path.join(args.dataset_root, "train_edges.pkl"))
    train_vertices_df = utils.read_pickle_from_file(os.path.join(args.dataset_root, "train_vertices.pkl"))
    train_vertices_df.drop(['id'], inplace=True, axis=1)
    train_labels = utils.create_user_labels_df(list(train_vertices_df.index), user_labels_dir)['label']
    utils.write_object_to_pickle_file('train_labels.pkl', train_labels)

    test_edges_df = utils.read_pickle_from_file(os.path.join(args.dataset_root, "test_edges.pkl"))
    test_vertices_df = utils.read_pickle_from_file(os.path.join(args.dataset_root, "test_vertices.pkl"))
    test_vertices_df.drop(['id'], inplace=True, axis=1)
    test_labels = utils.create_user_labels_df(list(test_vertices_df.index), user_labels_dir)['label']
    utils.write_object_to_pickle_file('test_labels.pkl', test_labels)

    print(f"Number of astroturfers on the train dataset {train_labels.sum()} ratio: {train_labels.sum()/len(train_labels)} (total: {len(train_labels)})")
    print(f"Number of astroturfers on the test dataset {test_labels.sum()} ratio: {test_labels.sum()/len(test_labels)} (total: {len(test_labels)})")

    target_encoding = preprocessing.LabelBinarizer()
    target_encoding.fit(train_labels)

    train_vertices_df, train_edges_df, train_inverse_node_mapping = reindex_edges(train_vertices_df, train_edges_df)
    test_vertices_df, test_edges_df, test_inverse_node_mapping = reindex_edges(test_vertices_df, test_edges_df)

    train_data = create_data(train_vertices_df, train_edges_df, train_labels, target_encoding)
    test_data = create_data(test_vertices_df, test_edges_df, test_labels, target_encoding)

    train_loader = DataLoader([train_data], batch_size=1, shuffle=True)
    test_loader = DataLoader([test_data], batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGENet(train_data.num_features, 32, 16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    def train():
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, _ = model(data.x, data.edge_index)
            loss = F.binary_cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate(loader):
        model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out, _ = model(data.x, data.edge_index)
                loss = F.binary_cross_entropy(out, data.y)
                total_loss += loss.item()
                pred = (out > 0.5).cpu().numpy()
                true = data.y.cpu().numpy()
                predictions.append(pred)
                true_labels.append(true)
        predictions = np.concatenate(predictions)
        true_labels = np.concatenate(true_labels)
        return total_loss / len(loader), predictions, true_labels

    for epoch in range(args.epochs):
        train_loss = train()
        val_loss, train_predictions, train_true_labels = evaluate(train_loader)
        _, val_predictions, val_true_labels = evaluate(test_loader)
        
        train_acc = accuracy_score(train_true_labels, train_predictions)
        val_acc = accuracy_score(val_true_labels, val_predictions)

        print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        if epoch == 0 or epoch % 15 == 0 or epoch == args.epochs -1:
            plot_tsne_node_embeddings(model, train_loader, device, epoch)  # Plot t-SNE of node embeddings after each epoch

    test_loss, test_predictions, test_true_labels = evaluate(test_loader)
    
    test_acc = accuracy_score(test_true_labels, test_predictions)
    test_precision = precision_score(test_true_labels, test_predictions)
    test_recall = recall_score(test_true_labels, test_predictions)
    test_f1 = f1_score(test_true_labels, test_predictions)
    
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Acc: {test_acc:.4f}')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')

    cm = confusion_matrix(test_true_labels, test_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoding.classes_)
    disp.plot()
    plt.savefig(f'tsne_social_graph/social_graph_confusion_matrix.png')
    plt.close()

    embeddings_lookup = {}
    for data, labels in zip([train_data, test_data], [train_labels, test_labels]):
        embeddings = infer(model, DataLoader([data], batch_size=1, shuffle=False))
        for i, index in enumerate(labels.index):
            embeddings_lookup[index] = embeddings[i].tolist()
    with open(os.path.join(args.dataset_root, 'users_graphsage_embeddings_lookup.json'), 'w') as f:
        json.dump(embeddings_lookup, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog="Example: python train_graphsage.py"
    )
    parser.add_argument(
        "--dataset-root",
        help="Directory of the dataset",
        dest="dataset_root",
        type=str,
        required=True
    )

    parser.add_argument(
        "--epochs",
        help="Number of epochs",
        dest="epochs",
        type=int,
        default=5
    )

    tsne_output_dir = "tsne_social_graph"
    if os.path.exists(tsne_output_dir) and os.path.isdir(tsne_output_dir):
        shutil.rmtree(tsne_output_dir)
    os.makedirs(tsne_output_dir)

    args = parser.parse_args()
    run(args)