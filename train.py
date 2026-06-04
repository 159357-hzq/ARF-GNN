from graph_data import GoTermDataset, collate_fn
from torch.utils.data import DataLoader
from network import CL_protNET
from nt_xent import NT_Xent
import torch.nn.functional as F
import torch
from sklearn import metrics
from utils import log
import argparse
from config import get_config
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def freeze_non_meta(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.gcn.arf_encoder.meta_learner.parameters():
        p.requires_grad = True

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True

def train(config, task, suffix):

    train_set = GoTermDataset("train", task, config.AF2model)
    valid_set = GoTermDataset("val", task, config.AF2model)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )


    output_dim = valid_set.y_true.shape[-1]

    model = CL_protNET(
        out_dim=output_dim,
        esm_embed=config.esmembed,
        pooling=config.pooling,
        pertub=config.contrast,
        max_hop=config.max_hop,
        dropout=config.dropout
    ).to(config.device)

    meta_params = list(model.gcn.arf_encoder.meta_learner.parameters())
    meta_param_ids = set(id(p) for p in meta_params)
    task_params = [p for p in model.parameters() if id(p) not in meta_param_ids]

    optimizer_task = torch.optim.Adam(
        params=task_params,
        lr=config.optimizer['lr'],
        weight_decay=config.optimizer.get('weight_decay', 0)
    )

    optimizer_meta = torch.optim.Adam(
        params=meta_params,
        lr=config.optimizer['lr'],
        weight_decay=config.optimizer.get('weight_decay', 0)
    )

    bce_loss = torch.nn.BCELoss(reduction='none')

    train_loss = []
    meta_loss_all = []
    val_loss = []
    val_aupr = []

    es = 0
    y_true_all = valid_set.y_true.float().reshape(-1)


    for ith_epoch in range(config.max_epochs):

        # =====================================================
        # 1) Task branch
        # =====================================================
        unfreeze_all(model)
        model.train()

        for idx_batch, batch in enumerate(train_loader):
            data = batch[0].to(config.device)
            y_true = batch[1].to(config.device)

            out = model(data, train_phase='Task')
            y_pred = out["y_pred"]

            task_loss = bce_loss(y_pred, y_true).mean()

            if config.contrast and "graph_feat_perturbed" in out:
                criterion = NT_Xent(out["graph_feat"].shape[0], 0.1, 1)
                cl_loss = 0.05 * criterion(out["graph_feat"], out["graph_feat_perturbed"])
                loss = task_loss + cl_loss
            else:
                loss = task_loss

            optimizer_task.zero_grad()
            loss.backward()
            optimizer_task.step()

            train_loss.append(loss.detach().cpu().numpy())
            if idx_batch % 20 == 0:
                log(f"{idx_batch}/{ith_epoch} train_epoch ||| Task Loss: {round(float(loss), 3)}")

        # =====================================================
        # 2) Meta branch
        # =====================================================
        freeze_non_meta(model)
        model.train()

        for idx_batch, batch in enumerate(train_loader):
            data = batch[0].to(config.device)
            y_true = batch[1].to(config.device)

            batch_size = y_true.size(0)
            forced_k = torch.randint(
                low=0,
                high=config.max_hop + 1,
                size=(batch_size, 1),
                device=config.device
            )

            out = model(data, train_phase='Meta', forced_k=forced_k)
            y_pred = out["y_pred"]
            meta_selected = out["meta_selected"]   # [B]

            with torch.no_grad():
                pred_bin = (y_pred >= 0.5).float()
                correctness = (pred_bin == y_true).float().mean(dim=1)  # [B]

            loss_meta = F.binary_cross_entropy(meta_selected, correctness)

            optimizer_meta.zero_grad()
            loss_meta.backward()
            optimizer_meta.step()

            meta_loss_all.append(loss_meta.detach().cpu().numpy())
            if idx_batch % 20 == 0:
                log(f"{idx_batch}/{ith_epoch} meta_epoch ||| Meta Loss: {round(float(loss_meta), 3)}")

        # =====================================================
        # 3) Validation
        # =====================================================
        model.eval()
        y_pred_all = []

        print("开始验证-------------")
        with torch.no_grad():
            for idx_batch, batch in enumerate(val_loader):
                data = batch[0].to(config.device)
                out = model(data, train_phase='Eval')
                y_pred = out["y_pred"]
                y_pred_all.append(y_pred)

            y_pred_all = torch.cat(y_pred_all, dim=0).cpu().reshape(-1)

            eval_loss = bce_loss(y_pred_all, y_true_all).mean()

            aupr = metrics.average_precision_score(
                y_true_all.numpy(),
                y_pred_all.numpy(),
                average="samples"
            )

            val_aupr.append(aupr)
            val_loss.append(eval_loss.cpu().numpy())
            
            log(f"{ith_epoch} VAL_epoch ||| loss: {round(float(eval_loss),3)} ||| aupr: {round(float(aupr),3)}")

            if ith_epoch == 0:
                best_eval_loss = eval_loss

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                es = 0
                torch.save(model.state_dict(), config.model_save_path + task + f"{suffix}.pt")
            else:
                es += 1
                print("Counter {} of 5".format(es))

            if es > 4:
                torch.save(
                    {
                        "train_bce": train_loss,
                        "meta_bce": meta_loss_all,
                        "val_bce": val_loss,
                        "val_aupr": val_aupr,
                    },
                    config.loss_save_path + task + f"{suffix}.pt"
                )
                break

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True' or v == 'true':
        return True
    if v == 'False' or v == 'false':
        return False

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=str, default='', choices=['bp', 'mf', 'cc'], help='')
    p.add_argument('--suffix', type=str, default='', help='')
    p.add_argument('--device', type=str, default='', help='')
    p.add_argument('--esmembed', default=True, type=str2bool, help='')
    p.add_argument('--pooling', default='MTP', type=str, choices=['MTP', 'GMP'], help='')
    p.add_argument('--contrast', default=True, type=str2bool, help='')
    p.add_argument('--AF2model', default=True, type=str2bool, help='')
    p.add_argument('--batch_size', type=int, default=64, help='')
    p.add_argument('--max_hop', type=int, default='', help='')
    p.add_argument('--dropout', type=float, default=0.2, help='')

    args = p.parse_args()
    config = get_config()

    config.optimizer['lr'] = 1e-4
    config.batch_size = args.batch_size
    config.max_epochs = 100
    config.max_hop = args.max_hop
    config.dropout = args.dropout

    if args.device != '':
        config.device = "cuda:" + args.device

    config.esmembed = args.esmembed
    config.pooling = args.pooling
    config.contrast = args.contrast
    config.AF2model = args.AF2model

    print(args)
    train(config, args.task, args.suffix)