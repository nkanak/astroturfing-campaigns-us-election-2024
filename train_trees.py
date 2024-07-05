#!/usr/bin/env python

import logging
import json
import os

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import shutil

from tqdm import tqdm

def tree_to_data(filename):
    logging.debug("Reading {}".format(filename))

    # read label
    with open(filename) as json_file:
        data = json.load(json_file)
        label = data["label"]
        # 0 = true, 1 = fake
        is_fake = label == "fake"

    vfeatures = []
    for node in data['nodes']:
        as_list = []
        for key in ["delay", "protected", "following_count", "listed_count", "statuses_count", "followers_count", "favourites_count", "verified"]:
            as_list.append(float(node[key]))
        as_list.extend(node["embedding"])
        vfeatures.append(as_list)

    vlabels = []
    vlabels.append(is_fake)

    edge_sources = []
    edge_targets = []
    for e in data['edges']:
        edge_sources.append(e['source'])
        edge_targets.append(e['target'])

    x = torch.tensor(vfeatures, dtype=torch.float)
    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    y = torch.tensor(vlabels, dtype=torch.long)
    result = Data(x=x, edge_index=edge_index, y=y)

    number_of_features = len(vfeatures[0])
    return label, number_of_features, result

from torch_geometric.nn.norm import BatchNorm
class Net(torch.nn.Module):
    def __init__(self, num_features):
        super(Net, self).__init__()
        self.batch_norm = BatchNorm(num_features)
        self.conv1 = GATConv(num_features, 32, heads=4, dropout=0.5)
        self.conv2 = GATConv(32 * 4, 16, heads=1, concat=False, dropout=0.5)  # Output size of 16
        self.linear = torch.nn.Linear(16, 1)  # Additional linear layer

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.batch_norm(x)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(x)
        x = self.linear(x)
        return x  # Return the linear layer output before applying sigmoid

def train(model, loader, device, optimizer, loss_op):
    model.train()
    total_loss = 0
    for data in tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()
        y_pred = torch.sigmoid(model(data)).squeeze()  # Apply sigmoid here for loss calculation
        y_true = data.y.float()
        loss = loss_op(y_pred, y_true)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    return total_loss / len(loader.dataset)

@torch.no_grad()
def test(model, loader, device):
    model.eval()
    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = torch.sigmoid(model(data.to(device)))  # Apply sigmoid here for prediction
        preds.append((out > 0.5).cpu().long())  # Apply threshold for binary classification
    y = torch.cat(ys, dim=0).numpy()
    pred = torch.cat(preds, dim=0).numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 'NaN'
    recall = tp / (tp + fn) if (tp + fn) > 0 else 'NaN'
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 'NaN'
    accuracy = accuracy_score(y, pred)
    return f1, precision, recall, accuracy

def plot_tsne(model, loader, device, epoch):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = model.batch_norm(x)
            x = F.elu(model.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=model.training)
            x = model.conv2(x, edge_index)
            x = global_mean_pool(x, batch)
            x = F.relu(x)  # ReLU activation
            embeddings.append(x.cpu().numpy())
            labels.append(data.y.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    tsne = TSNE(n_components=2, perplexity=10, learning_rate="auto", init="pca", random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings_2d[labels == 0, 0], embeddings_2d[labels == 0, 1], label='Real')
    plt.scatter(embeddings_2d[labels == 1, 0], embeddings_2d[labels == 1, 1], label='Fake')
    plt.legend()
    plt.title(f't-SNE of ReLU Outputs at Epoch {epoch}')
    plt.savefig(f'tsne_trees/tsne_epoch_{epoch}.png')
    plt.close()

def run(root_path):
    logging.info("Loading dataset")

    train_path = os.path.join(root_path, 'train')
    val_path = os.path.join(root_path, 'test')
    test_path = os.path.join(root_path, 'test')

    datasets = []
    for i, path in enumerate([train_path, val_path, test_path]):
        dataset_fake = []
        dataset_real = []
        for fentry in os.scandir(path):
            if fentry.path.endswith(".json") and fentry.is_file():
                label, number_of_features, t = tree_to_data(fentry.path)
                if label == "real":
                    dataset_real.append(t)
                else:
                    dataset_fake.append(t)

        number_of_samples = min(len(dataset_real), len(dataset_fake))
        #number_of_samples = 5000000
        dataset = dataset_real[:number_of_samples] + dataset_fake[:number_of_samples]
        datasets.append(dataset)

    # 0 = true, 1 = fake
    train_labels = [i.y.item() for i in datasets[0]]
    val_labels = [i.y.item() for i in datasets[1]]
    test_labels = [i.y.item() for i in datasets[2]]

    logging.info('Train dataset size: %s ' % len(train_labels))
    logging.info('Validation dataset size: %s ' % len(val_labels))
    logging.info('Test dataset size: %s' % len(test_labels))

    print('Number of fake news in train set: %s Number of real news: %s' % (len([i for i in train_labels if i == 1]), len([i for i in train_labels if i == 0])))
    print('Number of fake news in val set: %s Number of real news: %s' % (len([i for i in val_labels if i == 1]), len([i for i in val_labels if i == 0])))
    print('Number of fake news in test set: %s Number of real news: %s' % (len([i for i in test_labels if i == 1]), len([i for i in test_labels if i == 0])))

    train_loader = DataLoader(datasets[0], batch_size=32, shuffle=True)
    val_loader = DataLoader(datasets[1], batch_size=32, shuffle=False)
    test_loader = DataLoader(datasets[2], batch_size=4, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_features=number_of_features).to(device)
    loss_op = torch.nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

    # Start training
    NUM_OF_EPOCHS = 300
    for epoch in range(0, NUM_OF_EPOCHS):
        logging.info("Starting epoch {}".format(epoch))
        loss = train(model, train_loader, device, optimizer, loss_op)
        if epoch % 5 == 0:
            val_f1, val_precision, val_recall, val_accuracy = test(model, val_loader, device)
            print(f'Epoch: {epoch}, Loss: {loss}, Val F1: {val_f1} Val Prec: {val_precision} Val Rec: {val_recall} Val Acc: {val_accuracy}')
        if epoch == 0 or epoch % 5 == 0 or epoch == NUM_OF_EPOCHS -1:
            plot_tsne(model, test_loader, device, epoch)

    f1, precision, recall, accuracy = test(model, test_loader, device)
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall} F1: {f1}')
    return [accuracy, precision, recall, f1]

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )
    
    tsne_output_dir = "tsne_trees"
    if os.path.exists(tsne_output_dir) and os.path.isdir(tsne_output_dir):
        shutil.rmtree(tsne_output_dir)
    os.makedirs(tsne_output_dir)

    path = "produced_data/datasets/dataset"
    run(path)