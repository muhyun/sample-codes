import argparse
import os
import sys

import logging

import numpy as np
import torch as th
import torch.nn as nn

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

import dgl
from dgl.data.utils import load_graphs
from dgl.dataloading import NeighborSampler, ShaDowKHopSampler
from dgl.dataloading import NodeDataLoader

from transformer_utils import plot_roc, plot_p_r_curve, get_f1_score

from graphtransformer_models_node_edgefeature_version import graphtransformer_model_paper_mb

th.manual_seed(0)
np.random.seed(0)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# to find which edge feature contributes the model performance improvement
edge_exclude_lists = {
    'dev' : ['use', 'usedby'],
    'card': ['add', 'addedby', 'topup', 'topupby'],
    'order': ['buy', 'boughtby', 'pat', 'patedby', 'dat', 'datby']
}

def get_edge_exclude_list(edge_exc_level=0):
    '''
    001 - dev
    010 - card
    100 - order
    '''
    exc_list = []
    if edge_exc_level == 0:
        return exc_list
    
    if edge_exc_level & 1 > 0:
        exc_list = exc_list + edge_exclude_lists['dev']
        
    if edge_exc_level & 2 > 0:
        exc_list = exc_list + edge_exclude_lists['card']

    if edge_exc_level & 4 > 0:
        exc_list = exc_list + edge_exclude_lists['order']
        
    return exc_list

def refine_index(index_list, labels):
    refined_idx = []

    for idx in index_list.tolist():
        if labels[idx].item() in [0,1]:
            refined_idx.append(idx)
    
    refined_idx = th.Tensor(refined_idx).type(th.LongTensor)

    return refined_idx

def get_node_feature_dict(graph):
    print('loading node features', end='\r')
    node_features = {}
    node_features_dim = {}
    for ntype in graph.ntypes:
        if graph.nodes[ntype].data != {}:
            node_features[ntype] = graph.nodes[ntype].data[f'{ntype}_features'].float()
            node_features_dim[ntype] = node_features[ntype].shape[1]
            
    print('completed loading node features')
    return node_features, node_features_dim

def get_edge_feature_dict(graph, edge_exc_level=0):
    
    # for debug
    edge_exclude_list = get_edge_exclude_list(edge_exc_level)
    print('edge excl', edge_exclude_list)
    print('loading edge features', end='\r')
    edge_features = {}
    edge_features_dim = {}
    for c_etype in graph.canonical_etypes:
        if c_etype[1] in edge_exclude_list:
            print('skipping edge type', c_etype)
            continue
        if graph.edges[c_etype].data != {}:
            edge_features[c_etype] = graph.edges[c_etype].data['feature'].float()
            edge_features_dim[c_etype] = edge_features[c_etype].shape[1]
            
    print('completed loading edge features')
    return edge_features, edge_features_dim

def extract_node_feat(node_feat, input_nodes):
    feat = {}
    for ntype, nid in input_nodes.items():
        nid = input_nodes[ntype]
        if node_feat.get(ntype) is not None:
            feat[ntype] = node_feat[ntype][nid]
    return feat
    
def run_inference(model, val_dataloader, node_features, edge_feat_dict, labels, target, device, perm_edge=False):
    model.train(False)
    loss_fn = nn.CrossEntropyLoss()
    
    val_loss = []
    val_acc = []
    gt_list = []
    pred_list = []
    
    for input_n, output_n, mfgs in val_dataloader:
        if isinstance(mfgs, list) is True:
            mfgs = [m.to(device) for m in mfgs]
        else:
            mfgs = [mfgs.to(device) for l in range(gnn_n_layer)]
                    
        bt_in_feat = extract_node_feat(node_features, input_n)
        bt_in_feat = {k: v.to(device) for k, v in bt_in_feat.items()}
        
        bt_out_feat = extract_node_feat(node_features, output_n)
        bt_out_feat = {k: v.to(device) for k, v in bt_out_feat.items()}

        logits = model(mfgs, bt_in_feat, bt_out_feat, edge_feat_dict, perm_edge)
        
        bt_labels = labels[output_n[target]]
        bt_labels = bt_labels.to(device)
        
        loss = loss_fn(logits, bt_labels).sum()
        preds = logits.argmax(1)
        acc = (bt_labels == preds).float().mean()
        
        val_loss.append(loss.detach().cpu().numpy())
        val_acc.append(acc.cpu())
        
        gt_list += bt_labels.tolist()
        pred_list += preds.tolist()
        
    loss = np.array(val_loss)
    acc = np.array(val_acc)
    
    return loss.mean(), acc.mean(), gt_list, pred_list
        


def get_graph_data(ds_dir='/home/ec2-user/SageMaker/data_small', graph_fname='v2_nodefeature-graph.bin'):
    g_fname = os.path.join(ds_dir, graph_fname)

    if not os.path.exists(g_fname):
        s3_bucket = 'grab-aws-graphml-datadrop'
        object_name = f'data_small/{graph_fname}'
        print('download from s3', g_fname)
        download_file(s3_bucket, object_name, g_fname)

    print(f'Loading a graph from {g_fname}', end='\r')
    glist, label_dict = load_graphs(g_fname)
    print(f'Completed loading a graph from {g_fname}')

    g = glist[0]
    
    return g

def save_model(model, model_dir, epoch):
    logger.info('Saving the model.')
    
    if epoch is not None:
        model_fname = os.path.join(model_dir, f'model-epoch-{epoch}.pth')
    else:
        model_fname = os.path.join(model_dir, f'model.pth')
        
    th.save(model.cpu().state_dict(), model_fname)
    
    logger.info(f'Model saved to {model_fname}')
    
    return


def train(args):
    output_dirname = args.output_data_dir
    
    graph = get_graph_data(ds_dir=args.dataset_dir, graph_fname=args.graph_fname)
    print(graph)
    labels = graph.nodes['pax'].data['label']

    train_mask = graph.nodes['pax'].data['train_mask']
    val_mask = graph.nodes['pax'].data['val_mask']
    test_mask = graph.nodes['pax'].data['test_mask']

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    node_features, node_features_dim_dict = get_node_feature_dict(graph)
    edge_features, edge_features_dim_dict = get_edge_feature_dict(graph, args.edge_exc_level)

    train_idx = refine_index(train_idx, labels)
    val_idx = refine_index(val_idx, labels)
    test_idx = refine_index(test_idx, labels)

    target = 'pax'

    # define parameters
    input_dim = args.input_dim
    hid_dim = args.hid_dim
    output_dim = args.output_dim
    num_attn_head = args.num_attn_head
    gnn_n_layer = args.gnn_n_layer
    fanouts = args.fanouts

    attn_merge = args.attn_merge
    
    num_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    dropout = args.dropout
#     batch_norm = args.batch_norm
    
    perm_edge = args.perm_edge
    perm_edge_inf = args.perm_edge_inf
    
    embed_init = args.embed_init

    device = 'cuda:0' if th.cuda.is_available() else 'cpu'
    print('Using device: {}'.format(device))

    sampler = NeighborSampler([fanouts]*gnn_n_layer)

    dataloader = NodeDataLoader(graph,
                                {target: train_idx},
                                sampler,
                                batch_size=batch_size,
                                shuffle=True
                               )

    val_dataloader = NodeDataLoader(graph,
                                    {target: val_idx},
                                    sampler,
                                    batch_size=batch_size)

    test_dataloader = NodeDataLoader(graph,
                                    {target: test_idx},
                                    sampler,
                                    batch_size=batch_size)

    model = graphtransformer_model_paper_mb(input_dim,
                                  hid_dim,
                                  output_dim,
                                  num_attn_head,
                                  graph.ntypes,
                                  node_features_dim_dict,
                                  graph.canonical_etypes,
                                  edge_features_dim_dict,
                                  gnn_n_layer,
                                  attn_merge=attn_merge,
                                  dropout=dropout,
#                                   batch_norm=batch_norm,
                                  embed_init=embed_init,
                                  target=target,
                                  device=device)

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optim = th.optim.Adam(params=model.parameters(), lr=lr)

    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []

    gt_list = []
    pred_list = []

    for epoch in range(num_epochs):
        model.train()
        bt_loss = []
        bt_acc = []

        for cnt, (input_n, output_n, mfgs) in enumerate(dataloader):
            if isinstance(mfgs, list) is True:
                mfgs = [m.to(device) for m in mfgs]
            else:
                new_mfgs = [mfgs.to(device) for l in range(gnn_n_layer)]
                mfgs = new_mfgs

            bt_in_feat = extract_node_feat(node_features, input_n)
            bt_in_feat = {k: v.to(device) for k, v in bt_in_feat.items()}

            bt_out_feat = extract_node_feat(node_features, output_n)
            bt_out_feat = {k: v.to(device) for k, v in bt_out_feat.items()}

            logits = model(mfgs, bt_in_feat, bt_out_feat, edge_features, perm_edge)

            bt_labels = labels[output_n[target]]
            bt_labels = bt_labels.to(device)

            loss = loss_fn(logits, bt_labels).sum()
            preds = logits.argmax(1)
            acc = (bt_labels == preds).float().mean()
            
            # check if loss is NaN. If so, quit the training job
            if th.isnan(loss):
                print(f'>>>>>>>>>> Loss is NaN at batch count {cnt}. Abort the training. Loss {loss}')
                print('bt_labels', bt_labels)
                print('logits', logits)
                print('preds', preds)
                return epoch
            


            gt_list.append(bt_labels)
            pred_list.append(preds)

            bt_loss.append(loss.detach().cpu().numpy())
            bt_acc.append(acc.cpu())

            optim.zero_grad()
            
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            loss.backward()
            optim.step()

            if cnt % 10 == 0:
                if args.is_sagemaker is True:
                    print(f"In epoch: {epoch:4}; Batch cnt: {cnt:6}, Loss: {loss:.4f}, Acc:{acc:.4f}")
                else:
                    print(f"In epoch: {epoch:4}; Batch cnt: {cnt:6}, Loss: {loss:.4f}, Acc:{acc:.4f}", end='\r')

        loss = np.array(bt_loss)
        acc = np.array(bt_acc)

        val_loss, val_acc, _, _ = run_inference(model, val_dataloader, node_features, edge_features, labels, target, device, perm_edge_inf)

        print(f"In epoch: {epoch:4}; Tr_loss: {loss.mean():.4f}, Tr_acc:{acc.mean():.4f}, Val_loss: {val_loss:.4f}, Val_acc:{val_acc:.4f}")

        train_acc_list.append(acc.mean())
        train_loss_list.append(loss.mean())
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
    
    test_loss, test_acc, test_gt_list, test_pred_list = run_inference(model, test_dataloader, node_features, edge_features, labels, target, device, perm_edge_inf)

    print(f"Test dataset: Test_loss: {test_loss:.4f}, Test_acc:{test_acc:.4f}")
    
    if os.path.isdir(args.output_data_dir) is not True:
        os.makedirs(args.output_data_dir, exist_ok=True)
                    
    cf_fname = os.path.join(args.output_data_dir, 'cf.png')
    roc_fname = os.path.join(args.output_data_dir, 'roc.png')
    prcurve_fname = os.path.join(args.output_data_dir, 'prcurve.png')
    
    labels_names= ['non-fraud', 'fraud']

    print(classification_report(test_gt_list, test_pred_list, target_names=labels_names))

    cf = confusion_matrix(test_gt_list, test_pred_list, labels=[0,1])
    print(cf)

    df_cm = pd.DataFrame(cf, index = labels_names, columns = labels_names)
    plt.figure(figsize = (10,7))
    plt.ticklabel_format(style='plain', axis='y')
    sn.heatmap(df_cm, annot=True, fmt='d', annot_kws={'size': 16}, cmap="YlGnBu")
    plt.savefig(cf_fname)

    plot_roc(test_gt_list, test_pred_list, roc_fname)
    plot_p_r_curve(test_gt_list, test_pred_list, prcurve_fname)

    
#     test_labels = np.concatenate(mb_labels)
#     test_logits = np.concatenate(mb_logits)

#     plot_roc(test_labels, test_logits[:, 1])
#     plot_p_r_curve(test_labels, test_logits[:, 1])
    
    
    save_model(model, args.model_dir, num_epochs)
    
    return num_epochs


if __name__ =='__main__':

    if 'SM_MODEL_DIR' in os.environ:
        is_sagemaker = True
    else:
        is_sagemaker = False

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch-norm', default='False', choices=('True','False'))
    
    parser.add_argument('--input-dim', type=int, default=4)
    parser.add_argument('--hid-dim', type=int, default=4)
    parser.add_argument('--output-dim', type=int, default=2)
    parser.add_argument('--num-attn-head', type=int, default=4)
    parser.add_argument('--gnn-n-layer', type=int, default=3)
    parser.add_argument('--fanouts', type=int, default=5)
    
    parser.add_argument('--attn-merge', type=str, default='linear')
        
    parser.add_argument('--graph-fname', type=str, default='v4_nodeedgefeature_graph.bin')
    parser.add_argument('--perm-edge', default='False', choices=('True','False'))
    parser.add_argument('--perm-edge-inf', default='False', choices=('True','False'))
    
    parser.add_argument('--embed-init', type=str, default='constant')
    
    parser.add_argument('--edge-exc-level', type=int, default=0)
    
    # Data, model, and output directories
    if is_sagemaker is True:
        parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
        parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
        parser.add_argument('--dataset-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    else:
        parser.add_argument('--output-data-dir', type=str, default='./output')
        parser.add_argument('--model-dir', type=str, default='./models')
        parser.add_argument('--dataset-dir', type=str, default='/home/ec2-user/SageMaker/data_small')
    
    args, _ = parser.parse_known_args()
    
    args.perm_edge = True if args.perm_edge.lower() == 'true' else False
    args.perm_edge_inf = True if args.perm_edge_inf.lower() == 'true' else False
#     args.batch_norm = True if args.batch_norm.lower() == 'true' else False
                                   
    args.is_sagemaker = is_sagemaker
    print(args)
    ret = train(args)
    
    # if loss NaN occurs in the epoch 1, re-run the training.
    if ret == 0:
        print('>>>>>>>>>>> RE-RUN TRAINING FROM THE BEGINNING')
        train(args)