from graph_data import GoTermDataset, collate_fn
from torch.utils.data import DataLoader
from network import CL_protNET
import torch
from sklearn import metrics
import argparse
import pickle as pkl
from config import get_config
import numpy as np
from joblib import Parallel, delayed
import os
from utils import fmax, log
import my_evaluation

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

with open("ARF-main/data/ic_count.pkl", 'rb') as f:
    ic_count = pkl.load(f)

ic_count['bp'] = np.where(ic_count['bp'] == 0, 1, ic_count['bp'])
ic_count['mf'] = np.where(ic_count['mf'] == 0, 1, ic_count['mf'])
ic_count['cc'] = np.where(ic_count['cc'] == 0, 1, ic_count['cc'])

train_ic = {}
train_ic['bp'] = -np.log2(ic_count['bp'] / 69709)
train_ic['mf'] = -np.log2(ic_count['mf'] / 69709)
train_ic['cc'] = -np.log2(ic_count['cc'] / 69709)

def test(config, task, model_pt, test_type='test'):
    print(config.device)
    test_set = GoTermDataset(test_type, task)
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    output_dim = test_set.y_true.shape[-1]
    termIC = train_ic[task]

    model = CL_protNET(
        out_dim=output_dim,
        esm_embed=config.esmembed,
        pooling=config.pooling,
        pertub=False,
        max_hop=config.max_hop,
        dropout=config.dropout
    ).to(config.device)

    model.load_state_dict(torch.load(model_pt, map_location=config.device))
    model.eval()

    bce_loss = torch.nn.BCELoss()

    y_pred_all = []
    chosen_k_all = []
    meta_prob_all = []

    y_true_all = test_set.y_true.float()

    with torch.no_grad():
        for idx_batch, batch in enumerate(test_loader):
            data = batch[0].to(config.device)

            out = model(data, train_phase='Eval')

            y_pred = out["y_pred"]
            y_pred_all.append(y_pred.cpu())

            if "chosen_k" in out and out["chosen_k"] is not None:
                chosen_k_all.append(out["chosen_k"].cpu())

            if "meta_prob" in out and out["meta_prob"] is not None:
                meta_prob_all.append(out["meta_prob"].cpu())

        y_pred_all = torch.cat(y_pred_all, dim=0)

        eval_loss = bce_loss(y_pred_all, y_true_all)

        Fmax = my_evaluation.fmax(y_true_all.numpy(), y_pred_all.numpy(), task)
        aupr = my_evaluation.macro_aupr_test(y_true_all.numpy(), y_pred_all.numpy())
        Smin = my_evaluation.smin(termIC, y_true_all.numpy(), y_pred_all.numpy())

        log(
            f"Test ||| loss: {round(float(eval_loss), 3)} "
            f"||| aupr: {round(float(aupr), 3)} "
            f"||| Fmax: {round(float(Fmax), 3)} "
            f"||| Smin: {round(float(Smin), 3)}"
        )

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
    p.add_argument('--device', type=str, default='', help='')
    p.add_argument('--model', type=str, default='', help='')
    p.add_argument('--esmembed', default=True, type=str2bool, help='')
    p.add_argument('--pooling', default='MTP', type=str, choices=['MTP', 'GMP'],
                   help='Multi-set transformer pooling or Global mean pooling')
    p.add_argument('--AF2test', default=False, type=str2bool, help='')
    p.add_argument('--max_hop', type=int, default='', help='')
    p.add_argument('--dropout', type=float, default=0.2, help='')

    args = p.parse_args()
    print(args)

    config = get_config()
    config.batch_size = 32

    if args.device != '':
        config.device = "cuda:" + args.device

    config.esmembed = args.esmembed
    config.pooling = args.pooling
    config.max_hop = args.max_hop
    config.dropout = args.dropout

    if not args.AF2test:
        test(config, args.task, args.model)
    else:
        test(config, args.task, args.model, 'AF2test')