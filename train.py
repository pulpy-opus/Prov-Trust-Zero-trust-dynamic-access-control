##########################################################################################
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
##########################################################################################

import logging

from Prov_Trust_utils import *
from config import *
from model import *

# Setting for logging
logger = logging.getLogger("training_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'training.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def train_tgat(train_data,
              gnn,
              link_pred,
              optimizer,
              neighbor_loader):
    gnn.train()
    link_pred.train()
    neighbor_loader.reset_state()
    total_loss = 0
    num_events = train_data.src.size(0)
    for start in range(0, num_events, BATCH):
        end = min(start + BATCH, num_events)
        optimizer.zero_grad()
        src = train_data.src[start:end]
        pos_dst = train_data.dst[start:end]
        t = train_data.t[start:end]
        msg = train_data.msg[start:end]
        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        x = train_data.msg[n_id, :node_state_dim]
        last_update = train_data.t[n_id]
        z = gnn(x, last_update, edge_index, train_data.t[e_id], train_data.msg[e_id])
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        y_pred = torch.cat([pos_out], dim=0)
        y_true = []
        for m in msg:
            l = tensor_find(m[node_embedding_dim:-node_embedding_dim], 1) - 1
            y_true.append(l)
        y_true = torch.tensor(y_true).to(device=device)
        y_true = y_true.reshape(-1).to(torch.long).to(device=device)
        loss = criterion(y_pred, y_true)
        neighbor_loader.insert(src, pos_dst)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * (end - start)
    return total_loss / num_events

def load_train_data():
    graph_4_2 = torch.load(graphs_dir + "/graph_4_2.TemporalData.simple").to(device=device)
    graph_4_3 = torch.load(graphs_dir + "/graph_4_3.TemporalData.simple").to(device=device)
    graph_4_4 = torch.load(graphs_dir + "/graph_4_4.TemporalData.simple").to(device=device)
    return [graph_4_2, graph_4_3, graph_4_4]

def init_tgat_models(node_feat_size):
    # Use TimeEncode within GraphAttentionEmbedding for temporal encoding
    gnn = GraphAttentionEmbedding(
        in_channels=node_state_dim,
        out_channels=edge_dim,
        msg_dim=node_feat_size,
        time_enc=TimeEncode(time_dim)
    ).to(device)
    out_channels = len(include_edge_type)
    link_pred = LinkPredictor(in_channels=edge_dim, out_channels=out_channels).to(device)
    optimizer = torch.optim.Adam(
        list(gnn.parameters()) + list(link_pred.parameters()), lr=lr, eps=eps, weight_decay=weight_decay)
    neighbor_loader = LastNeighborLoader(max_node_num, size=neighbor_size, device=device)
    return gnn, link_pred, optimizer, neighbor_loader

if __name__ == "__main__":
    logger.info("Start logging.")
    train_data = load_train_data()
    node_feat_size = train_data[0].msg.size(-1)
    gnn, link_pred, optimizer, neighbor_loader = init_tgat_models(node_feat_size=node_feat_size)
    for epoch in tqdm(range(1, epoch_num+1)):
        for g in train_data:
            loss = train_tgat(
                train_data=g,
                gnn=gnn,
                link_pred=link_pred,
                optimizer=optimizer,
                neighbor_loader=neighbor_loader
            )
            logger.info(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')
    # Save model
    model = [gnn, link_pred, neighbor_loader]
    os.system(f"mkdir -p {models_dir}")
    torch.save(model, f"{models_dir}/models.pt")
