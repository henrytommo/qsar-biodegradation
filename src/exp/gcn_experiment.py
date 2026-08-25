"""
gcn setup. node features come from RDKit-parsed SMILES graphs
plus tabular QSAR descriptors (incl. the HOMO-LUMO gap)
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from exp.spec import ModelSpec

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
def predict_raw(model, graphs, batch_size=64):
    """(y_true, preds, proba) on graphs -- the generic driver turns these into metrics."""
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    model.eval()
    _, logits, y = _run_epoch(model, loader, pos_weight=torch.tensor(1.0))
    proba = torch.sigmoid(logits).numpy()
    preds = (proba >= 0.5).astype(int)
    return y.numpy(), preds, proba


def _build(params, dims, random_state):
    return _make_model(params, dims["node_feat_dim"], dims["global_feat_dim"], random_state=random_state)


def _fit(model, train, val, params):
    model, best_epoch = fit(model, train.data, val.data, params)
    return model, {"best_epoch": best_epoch}


SPEC = ModelSpec(
    name="gcn",
    param_distributions=PARAM_DISTRIBUTIONS,
    graph=True,
    build=_build,
    fit=_fit,
    predict=lambda model, ds: predict_raw(model, ds.data),
    ranking=None,   # no native ranking; feature-count validation ranks via XGBoost
)
