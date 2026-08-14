"""GCN experiment: random hyperparameter search over CV folds, then a final fit/eval
on the held-out test set. Node features come from RDKit-parsed SMILES graphs; the
tabular QSAR descriptors are concatenated in as global features after graph pooling.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from de import gcn_preprocessing as gcnp
from utils.metrics import aggregate_metrics, classification_metrics
from utils.search import random_search

PARAM_DISTRIBUTIONS = {
    "hidden_dim": [32, 64, 128],
    "n_gcn_layers": [2, 3],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [0.001, 0.005, 0.01],
    "weight_decay": [0.0, 1e-4, 1e-3],
    "batch_size": [32, 64],
}

MAX_EPOCHS = 150
PATIENCE = 15


class GCN(nn.Module):
    def __init__(self, node_feat_dim, global_feat_dim, hidden_dim=64, n_gcn_layers=2, dropout=0.2):
        super().__init__()
        dims = [node_feat_dim] + [hidden_dim] * n_gcn_layers
        self.convs = nn.ModuleList(GCNConv(dims[i], dims[i + 1]) for i in range(n_gcn_layers))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_embed = global_mean_pool(x, batch)
        combined = torch.cat([graph_embed, data.global_feats], dim=1)
        return self.head(combined).squeeze(-1)


def _make_model(params, node_feat_dim, global_feat_dim, random_state=1):
    torch.manual_seed(random_state)
    return GCN(node_feat_dim, global_feat_dim, hidden_dim=params["hidden_dim"],
               n_gcn_layers=params["n_gcn_layers"], dropout=params["dropout"])


def _pos_weight(graphs):
    y = torch.cat([g.y for g in graphs])
    return (len(y) - y.sum()) / y.sum()


def _run_epoch(model, loader, pos_weight, optimizer=None):
    train = optimizer is not None
    model.train(train)
    all_logits, all_y = [], []
    total_loss = 0.0

    with torch.set_grad_enabled(train):
        for batch in loader:
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, batch.y, pos_weight=pos_weight)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            all_logits.append(logits.detach())
            all_y.append(batch.y)

    return total_loss / len(loader.dataset), torch.cat(all_logits), torch.cat(all_y)


def fit(model, train_graphs, val_graphs, params, max_epochs=MAX_EPOCHS, patience=PATIENCE, random_state=1):
    """Train with early stopping on val loss; restores the model to its best epoch."""
    torch.manual_seed(random_state)
    train_loader = DataLoader(train_graphs, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    pos_weight = _pos_weight(train_graphs)

    best_val_loss, best_state, epochs_no_improve, best_epoch = float("inf"), None, 0, 0

    for epoch in range(max_epochs):
        _run_epoch(model, train_loader, pos_weight, optimizer)
        val_loss, _, _ = _run_epoch(model, val_loader, pos_weight)

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, epochs_no_improve = val_loss, epoch + 1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch


@torch.no_grad()
def evaluate(model, graphs, batch_size=64):
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    model.eval()
    _, logits, y = _run_epoch(model, loader, pos_weight=torch.tensor(1.0))
    proba = torch.sigmoid(logits).numpy()
    preds = (proba >= 0.5).astype(int)
    return classification_metrics(y.numpy(), preds, proba)


def _score_params(params, method, random_state, graphs, node_feat_dim, global_feat_dim, n_splits=5):
    fold_metrics = []
    for train_graphs, val_graphs in gcnp.cv_folds(graphs, method=method, n_splits=n_splits, random_state=random_state):
        model = _make_model(params, node_feat_dim, global_feat_dim, random_state=random_state)
        model, best_epoch = fit(model, train_graphs, val_graphs, params, random_state=random_state)

        metrics = evaluate(model, val_graphs)
        metrics["best_epoch"] = best_epoch
        fold_metrics.append(metrics)

    return {"params": params, "method": method, **aggregate_metrics(fold_metrics)}


def fit_final_model(train_graphs, test_graphs, params, node_feat_dim, global_feat_dim,
                     method="standardize", val_size=0.15, random_state=1):
    """Refit with a small internal val slice for early stopping, evaluate once on test_graphs."""
    tr_graphs, val_graphs = gcnp.holdout_split(train_graphs, test_size=val_size, random_state=random_state)
    tr_scaled, val_scaled, scaler = gcnp.scale_graph_features(tr_graphs, val_graphs, method=method)
    test_scaled = gcnp.transform_graph_features(test_graphs, scaler)

    model = _make_model(params, node_feat_dim, global_feat_dim, random_state=random_state)
    model, _ = fit(model, tr_scaled, val_scaled, params, random_state=random_state)

    return model, evaluate(model, test_scaled), test_scaled


if __name__ == "__main__":
    from utils.logger import Logger

    log = Logger().log

    graphs, n_skipped = gcnp.load_graph_dataset()
    log(f"Loaded {len(graphs)} graphs ({n_skipped} skipped)")

    node_feat_dim = graphs[0].x.shape[1]
    global_feat_dim = graphs[0].global_feats.shape[1]
    train_graphs, test_graphs = gcnp.holdout_split(graphs)

    log(f"Random search: {len(PARAM_DISTRIBUTIONS)} hyperparams, 5-fold CV, train graphs {len(train_graphs)}")
    results = random_search(_score_params, PARAM_DISTRIBUTIONS, n_iter=25,
                             graphs=train_graphs, node_feat_dim=node_feat_dim,
                             global_feat_dim=global_feat_dim, n_splits=5)

    best = results[0]
    log(f"\nBest CV params: {best['params']}")
    log(f"CV F1:      {best['f1_mean']:.3f} +/- {best['f1_std']:.3f}")
    log(f"CV ROC-AUC: {best['roc_auc_mean']:.3f} +/- {best['roc_auc_std']:.3f}")
    log(f"CV PR-AUC:  {best['pr_auc_mean']:.3f} +/- {best['pr_auc_std']:.3f}")

    model, metrics, test_scaled = fit_final_model(train_graphs, test_graphs, best["params"],
                                                    node_feat_dim, global_feat_dim)
    log(f"\nFinal test metrics: {({k: round(v, 3) for k, v in metrics.items()})}")
