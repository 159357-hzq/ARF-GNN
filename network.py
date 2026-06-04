import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from pool import GraphMultisetTransformer

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.mlp(x)

class SoftMetaLearner(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(SoftMetaLearner, self).__init__()
        self.dropout = dropout
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, hop_feats):
        if self.dropout is not None and self.dropout > 0:
            hop_feats = F.dropout(hop_feats, p=self.dropout, training=self.training)

        out, _ = self.rnn(hop_feats)
        logits = self.scorer(out).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs

class ARFEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, max_hop=4, dropout=0.2):
        super(ARFEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_hop = max_hop
        self.dropout = dropout

        self.backbone1 = nn.Linear(input_dim, hidden_dim)
        self.backbone2 = nn.Linear(hidden_dim, hidden_dim)

        self.hop_convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, add_self_loops=True, bias=True)
            for _ in range(max_hop)
        ])
        self.hop_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(max_hop)
        ])

        self.self_proj = nn.Linear(hidden_dim, hidden_dim)

        self.meta_learner = SoftMetaLearner(hidden_dim, dropout=dropout)

        self.hop_att = nn.Linear(hidden_dim, 1)

        self.drop = nn.Dropout(dropout)

    def _build_multi_hop_feats(self, x, edge_index):
        x = self.drop(x)
        x = F.relu(self.backbone1(x))
        x = self.drop(x)
        x = F.relu(self.backbone2(x))

        hop_feats = []
        h = x

        z0 = self.self_proj(h)
        hop_feats.append(z0)

        for i, conv in enumerate(self.hop_convs):
            h_new = conv(h, edge_index)
            h_new = F.relu(h_new)
            h = self.hop_norms[i](h + h_new)
            hop_feats.append(h)

        # [N, K+1, D]
        hop_feats = torch.stack(hop_feats, dim=1)
        return hop_feats

    def _pool_hop_feats_to_graph(self, hop_feats, batch):
        """
        hop_feats: [N, K+1, D]
        return: [B, K+1, D]
        """
        graph_hops = []
        num_hops = hop_feats.size(1)
        for i in range(num_hops):
            graph_hops.append(gap(hop_feats[:, i, :], batch))
        return torch.stack(graph_hops, dim=1)

    def forward(self, x, edge_index, batch, train_phase='Task', forced_k=None):

        hop_feats = self._build_multi_hop_feats(x, edge_index)
        graph_hop_feats = self._pool_hop_feats_to_graph(hop_feats, batch)

        meta_prob = self.meta_learner(graph_hop_feats)

        B, K1 = meta_prob.size()

        if train_phase in ['Task', 'Eval']:
            if forced_k is None:
                chosen_k = meta_prob.argmax(dim=1, keepdim=True)
            else:
                chosen_k = forced_k
            meta_selected = meta_prob.gather(1, chosen_k).squeeze(-1)

        elif train_phase == 'Meta':
            if forced_k is None:
                chosen_k = torch.randint(
                    low=0,
                    high=K1,
                    size=(B, 1),
                    device=x.device
                )
            else:
                chosen_k = forced_k
            meta_selected = meta_prob.gather(1, chosen_k).squeeze(-1)
        else:
            raise ValueError(f"Unsupported train_phase: {train_phase}")

        node_k = chosen_k[batch]

        hop_ids = torch.arange(K1, device=x.device).view(1, K1)
        valid_mask = hop_ids <= node_k

        hop_logits = self.hop_att(hop_feats).squeeze(-1)
        hop_logits = hop_logits.masked_fill(~valid_mask, -1e9)
        hop_alpha = torch.softmax(hop_logits, dim=1)

        out = (hop_feats * hop_alpha.unsqueeze(-1)).sum(dim=1)

        return {
            "node_feat": out,
            "hop_feats": hop_feats,
            "graph_hop_feats": graph_hop_feats,
            "meta_prob": meta_prob,
            "meta_selected": meta_selected,
            "chosen_k": chosen_k,
            "hop_alpha": hop_alpha
        }

class GraphCNN(nn.Module):

    def __init__(self,
                 pooling='MTP',
                 max_hop=4,
                 dropout=0.2):
        super(GraphCNN, self).__init__()

        self.pooling = pooling
        self.drop1 = nn.Dropout(p=dropout)

        self.arf_encoder = ARFEncoder(
            input_dim=512,
            hidden_dim=512,
            max_hop=max_hop,
            dropout=dropout
        )

        if self.pooling == "MTP":
            self.pool = GraphMultisetTransformer(
                512, 256, 512, None, 10000, 0.25,
                ['GMPool_G', 'GMPool_G'],
                num_heads=8,
                layer_norm=True
            )
        else:
            self.pool = gmp

        self.last_meta_prob = None
        self.last_meta_selected = None
        self.last_chosen_k = None
        self.last_hop_alpha = None
        self.last_graph_hop_feats = None

    def forward(self, x, data, train_phase='Task', forced_k=None, pertubed=False):
        x = self.drop1(x)

        arf_out = self.arf_encoder(
            x=x,
            edge_index=data.edge_index.long(),
            batch=data.batch,
            train_phase=train_phase,
            forced_k=forced_k
        )

        h = arf_out["node_feat"]

        self.last_meta_prob = arf_out["meta_prob"]
        self.last_meta_selected = arf_out["meta_selected"]
        self.last_chosen_k = arf_out["chosen_k"]
        self.last_hop_alpha = arf_out["hop_alpha"]
        self.last_graph_hop_feats = arf_out["graph_hop_feats"]

        if pertubed:
            random_noise = torch.rand_like(h).to(h.device)
            h = h + torch.sign(h) * F.normalize(random_noise, dim=-1) * 0.05

        if self.pooling == 'MTP':
            g_level_feat = self.pool(h, data.batch, data.edge_index.long())
        else:
            g_level_feat = self.pool(h, data.batch)

        return {
            "node_feat": h,
            "graph_feat": g_level_feat,
            "meta_prob": arf_out["meta_prob"],
            "meta_selected": arf_out["meta_selected"],
            "chosen_k": arf_out["chosen_k"]
        }

class CL_protNET(nn.Module):
    def __init__(self,
                 out_dim,
                 esm_embed=True,
                 pooling='MTP',
                 pertub=False,
                 rw_dim=None,
                 max_hop=4,
                 dropout=0.2):
        super(CL_protNET, self).__init__()
        self.esm_embed = esm_embed
        self.pertub = pertub
        self.out_dim = out_dim
        self.pooling = pooling

        self.one_hot_embed = nn.Embedding(21, 96)
        self.proj_aa = nn.Linear(96, 512)

        if esm_embed:
            self.proj_esm = nn.Linear(1280, 512)

        self.use_rw = rw_dim is not None
        if self.use_rw:
            self.proj_rw = nn.Linear(rw_dim, 512)
            self.fuse_proj = nn.Linear(1024, 512)

        self.gcn = GraphCNN(
            pooling=pooling,
            max_hop=max_hop,
            dropout=dropout
        )

        self.readout = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, out_dim),
            nn.Sigmoid()
        )

    def _encode_input(self, data):
        x_aa = self.one_hot_embed(data.native_x.long())
        x_aa = self.proj_aa(x_aa)

        if self.esm_embed:
            x_esm = self.proj_esm(data.x.float())
            x_seq = F.relu(x_aa + x_esm)
        else:
            x_seq = F.relu(x_aa)

        if self.use_rw and hasattr(data, "rw_feat"):
            x_rw = self.proj_rw(data.rw_feat.float())
            x = torch.cat([x_seq, x_rw], dim=-1)
            x = F.relu(self.fuse_proj(x))
        else:
            x = x_seq

        return x

    def forward(self, data, train_phase='Task', forced_k=None):
        x = self._encode_input(data)

        out1 = self.gcn(
            x, data,
            train_phase=train_phase,
            forced_k=forced_k,
            pertubed=False
        )
        y_pred = self.readout(out1["graph_feat"])

        result = {
            "y_pred": y_pred,
            "graph_feat": out1["graph_feat"],
            "meta_prob": out1["meta_prob"],
            "meta_selected": out1["meta_selected"],
            "chosen_k": out1["chosen_k"]
        }

        if self.pertub and train_phase == 'Task':
            out2 = self.gcn(
                x, data,
                train_phase=train_phase,
                forced_k=forced_k,
                pertubed=True
            )
            result["graph_feat_perturbed"] = out2["graph_feat"]

        return result