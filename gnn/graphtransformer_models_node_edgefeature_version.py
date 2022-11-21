import dgl
import torch as th
import torch.nn.functional as F

from dgl import function as fn
from dgl.nn.functional import edge_softmax

import dgl.function as dglf
import torch.nn as nn

print(dgl.__version__)
print(th.__version__)

class graphtransformer_input_layer_mb(nn.Module):
    """
    This layer model works for input layer, including the input feature combination and message passing with attention.
    2021/03/31: For mini-batch, the init fn does not change, only the forward fn need to change.
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_attn_head,
                 ntype_list,
                 ntype_feature_dim_dict,
                 can_etype_list,
                 etype_feature_dim_dict,
                 inner_type_agg_fn='mean',
                 inter_type_agg_fn='sum',
                 attn_merge='linear',
                 dropout=None,
                 device='cpu'):
        super(graphtransformer_input_layer_mb, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_attn_head = num_attn_head
        self.ntype_list = ntype_list
        self.ntype_feature_dim_dict = ntype_feature_dim_dict
        self.can_etype_list = can_etype_list
        self.etype_feature_dim_dict = etype_feature_dim_dict
        self.inner_type_agg_fn = inner_type_agg_fn
        self.inter_type_agg_fn = inter_type_agg_fn
        self.device = device
#         self.batch_norm = batch_norm

        # Grab POC: to apply Linear transformation to node features to make their dimensions the same as input_dim
        self.node_feature_linears = {}
        for ntype in self.ntype_list:
            if ntype in self.ntype_feature_dim_dict:
                self.node_feature_linears[ntype] = nn.Linear(self.ntype_feature_dim_dict[ntype], self.in_dim).to(self.device)
                self._norm_init_linear(self.node_feature_linears[ntype])
                
        self.edge_feature_linears = {}
        for c_etype in self.can_etype_list:
            if c_etype in self.etype_feature_dim_dict:
                self.edge_feature_linears[c_etype] = nn.Linear(self.etype_feature_dim_dict[c_etype], self.in_dim).to(self.device)
                self._norm_init_linear(self.edge_feature_linears[c_etype])
                
        self.edge_attn_linear = nn.Linear(self.in_dim, self.num_attn_head)
        self._norm_init_linear(self.edge_attn_linear)
        
        self.attn_merge = attn_merge
        self.attn_merger_linear = nn.Linear(self.num_attn_head * 2, self.num_attn_head)
        self._norm_init_linear(self.attn_merger_linear)
            
        # Set up the 3 linear transformation weights: query, key, and value for dst, src, src node type
        query_weight = {}
        key_weight = {}
        value_weight = {}
        for src_type, e_type, dst_type in self.can_etype_list:
            # print(src_type, e_type, dst_type)
            if query_weight.get(dst_type) is None:
                query_weight[dst_type] = nn.Linear(self.in_dim, self.out_dim * self.num_attn_head)
            if key_weight.get(src_type) is None:
                key_weight[src_type] = nn.Linear(self.in_dim, self.out_dim * self.num_attn_head)
            if value_weight.get(src_type) is None:
                value_weight[src_type] = nn.Linear(self.in_dim, self.out_dim * self.num_attn_head)

        self.query_weight = nn.ModuleDict(query_weight)
        self.key_weight = nn.ModuleDict(key_weight)
        self.value_weight = nn.ModuleDict(value_weight)
        
        # Set up attention weights for each node type
        self.node_attn_weight = nn.ModuleDict()
        for i in range(len(self.ntype_list)):
            self.node_attn_weight[self.ntype_list[i]] = nn.Linear(self.out_dim, 1, bias=False)

        # Set up droput
        if dropout is not None and dropout > 0:
            self.layer_dropout = nn.Dropout(dropout)
        else:
            self.layer_dropout = None

        # Set up layer normalization
        self.layer_norm = nn.LayerNorm(self.num_attn_head * self.out_dim)

        # Initialize all weights with uniform distribution 
        for _, linear_weight in self.query_weight.items():
            self._uniform_init_linear(linear_weight)
        for _, linear_weight in self.key_weight.items():
            self._uniform_init_linear(linear_weight)
        for _, linear_weight in self.value_weight.items():
            self._uniform_init_linear(linear_weight)
        for _, linear_weight in self.node_attn_weight.items():
            self._uniform_init_linear(linear_weight)

    def _uniform_init_linear(self, linear):
        nn.init.uniform_(linear.weight)
        if linear.bias is not None:
            nn.init.constant_(linear.bias, 0)
            
    def _norm_init_linear(self, linear):
        nn.init.normal_(linear.weight, std=0.01)
        
    def forward(self, graph, node_feats_dict, node_emb_dict, edge_emb_dict, edge_feats_dict, perm_edge=False, ignore_self_loop=True, debug=False):
        """
        Support DGL mini-batch mode.

        :param graph: an MFG in hetero mode
        :param node_feats_dict: A dictionary including both src and dst nodes, if src and dst nodes are in the same type,
                                the first set of nodes are dst nodes.
        :param node_emb_dict: Same as the node_feat_dict, but including embedding only.
        :param edge_emb_dict: A dictionary, including type-wise embedding of edges
        :return:
        """
        if graph.is_block or True:

            etype_fn = {}
            for src_type, e_type, dst_type in graph.canonical_etypes:
                c_e_type = (src_type, e_type, dst_type)
                
                if src_type == dst_type and ignore_self_loop is True:
                    continue
                    
                # extract a bipartite
                subgraph = graph[src_type, e_type, dst_type]
                
                # check if no edges
                if subgraph.num_edges() == 0:
                    continue
    
                if debug is True:
                    print('input layer ', c_e_type)

                # Compute target input hidden values 
                if node_feats_dict.get(dst_type) is not None:                    
                    # Grab POC: applying Linear transformation to make the dim the same as input dimension (i.ie in_dim)
                    X = node_feats_dict.get(dst_type)[ :subgraph.number_of_dst_nodes()]                        
                    X = self.node_feature_linears[dst_type](X)                        
                    dst_feat = X + edge_emb_dict[e_type]                    
                else:
                    dst_feat = th.zeros(subgraph.number_of_dst_nodes(), self.in_dim).to(self.device) + edge_emb_dict[e_type]
                
                # Compute source input hidden values 
                if node_feats_dict.get(src_type) is not None:
                    
                    X = node_feats_dict.get(src_type)
                    X = self.node_feature_linears[src_type](X)
#                     if src_type in self.ntype_feature_dim_dict:                        
#                         X = self.node_feature_linears[src_type](X)
                    
                    src_feat = X + node_emb_dict[src_type] + edge_emb_dict[e_type]
                else:
                    src_feat = th.zeros(subgraph.number_of_src_nodes(src_type), self.in_dim).to(self.device) + \
                               node_emb_dict[src_type] + edge_emb_dict[e_type]
                
                # Grab POC: using edge feature
                ori_eid = subgraph.edata[dgl.EID]

                if debug is True:
                    print('\n', c_e_type)
                    print(subgraph)
                    print('subgraph.number_of_src_nodes(src_type)', subgraph.number_of_src_nodes(src_type))
                    print('src node', subgraph.srcnodes[src_type].data[dgl.NID].shape[0])
                    print(subgraph.srcnodes[src_type])
                    print('edge    ', subgraph.num_edges())
                    print(subgraph.edges())
                    print('src_feat ', src_feat.shape)
                
                if edge_feats_dict.get((src_type, e_type, dst_type)) is not None:
                    edge_feat = edge_feats_dict[(src_type, e_type, dst_type)][ori_eid].to(self.device)
                    
                    # randomize the edge features to make noisy
                    if perm_edge is True:
                        perm_idx = th.randperm(edge_feat.shape[0])
                        edge_feat = edge_feat[perm_idx]
                        
                    edge_feat = self.edge_feature_linears[c_e_type](edge_feat)   
                else:
                    edge_feat = edge_emb_dict[e_type].to(self.device)                    # This is just 1d type embedding
                    edge_feat = th.stack([edge_feat] * subgraph.num_edges())    # extend to all edges

            
                # Grab: Get attention using edge feature
                subgraph.srcdata['h_s'] = src_feat
                subgraph.edata['f_e'] = edge_feat
                subgraph.apply_edges(fn.u_add_e('h_s', 'f_e', 'h_add'))
                temp_attn = subgraph.edata['h_add']
                temp_attn_sfm = edge_softmax(subgraph, temp_attn)
                temp_attn_sfm = self.edge_attn_linear(temp_attn_sfm)
                temp_attn_sfm = temp_attn_sfm.unsqueeze(dim=2)
                subgraph.edata['temp_attn'] = temp_attn_sfm
                                
                # Step 1: 
                query = self.query_weight[dst_type]
                key = self.key_weight[src_type]
                value = self.value_weight[src_type]

                # expend to 3D
                query_dst = query(dst_feat).view(-1, self.num_attn_head, self.out_dim)
                key_src = key(src_feat).view(-1, self.num_attn_head, self.out_dim)
                value_src = value(src_feat).view(-1, self.num_attn_head, self.out_dim)

                # Step 2: compute attention values of key and query, and then put them to edges for message passing
                attn_query_dst = self.node_attn_weight[dst_type](query_dst)
                attn_key_src = self.node_attn_weight[src_type](key_src)

                if debug is True:
                    print('B edge feat.        ', edge_feat.shape)
                    print('B subgraph temp_attn    ', subgraph.edata['temp_attn'].shape)
                    print('B dst attn_query_dst', attn_query_dst.shape)
                    print('B src attn_key_src  ', attn_key_src.shape)
                    print('B src value_src     ', value_src.shape)

                # put to nodes data field first
                subgraph.nodes[dst_type].data['query'] = attn_query_dst
                subgraph.nodes[src_type].data['key'] = attn_key_src
                subgraph.nodes[src_type].data['value'] = value_src
                
                if debug is True:
                    print('A edge feat.        ', edge_feat.shape)
                    print('A dst attn_query_dst', attn_query_dst.shape)
                    print('A src attn_key_src  ', attn_key_src.shape)
                    print('A src value_src     ', value_src.shape)

                # Step 3: compute attention along edges. 
                subgraph.apply_edges(dglf.u_add_v('key', 'query', 'attn_head'))
                edge_heads = subgraph.edges[e_type].data['attn_head']
                edge_heads = edge_heads / (self.out_dim**0.5)

                # Step 4: do softmax. 
                # edge_heads = th.squeeze(edge_heads)
                edge_heads = edge_heads.view(edge_heads.shape[:-1])
            
                # Grab: adding a linear layer (input - concat(edge_heads, edge_feat), output - tensor of shape edge_head.shape)
                self_attn = edge_softmax(subgraph, edge_heads)

                # Step 5: compute message. 
                if self.layer_dropout is not None:
                    self_heads = self.layer_dropout(self_attn)
                else:
                    self_heads = self_attn

                self_heads = th.unsqueeze(self_heads, dim=-1)
                
                # grab : combine two attns (design choices: dot prod, max, mean, linear, ....)
                if self.attn_merge == 'linear':
                    head_cat = th.cat((self_heads, temp_attn_sfm), 1)
                    head_cat = th.squeeze(head_cat,2 )
                    self_heads = self.attn_merger_linear(head_cat)
                    self_heads = th.unsqueeze(self_heads, dim=-1)
                elif self.attn_merge == 'mul':
                    self_heads = self_heads * temp_attn_sfm
                elif self.attn_merge == 'add':
                    self_heads = self_heads + temp_attn_sfm
                
                subgraph.edges[e_type].data['a_head'] = self_heads
                subgraph.apply_edges(lambda edges: {'attn_msg': edges.src['value'] * edges.data['a_head']})

                # Step 6: aggregation all edges into target nodes. 
                if self.inner_type_agg_fn == 'mean':
                    etype_fn[c_e_type] = (dglf.copy_e('attn_msg', 'm'), dglf.mean('m', 'h'))
                elif self.inner_type_agg_fn == 'sum':
                    etype_fn[c_e_type] = (dglf.copy_e('attn_msg', 'm'), dglf.sum('m', 'h'))
                elif self.inner_type_agg_fn == 'max':
                    etype_fn[c_e_type] = (dglf.copy_e('attn_msg', 'm'), dglf.max('m', 'h'))
                else:
                    raise KeyError('Inner type aggregator type {} not recognized.'.format(self.inner_type_agg_fn))

            # Step 7: Aggregate all different relations
            if self.inter_type_agg_fn in ['sum', 'mean', 'max']:
                graph.multi_update_all(etype_fn, self.inter_type_agg_fn)
            else:
                raise KeyError('Inter type aggregator type {} not recognized.'.format(self.inter_type_agg_fn))

            output = {}
            for n_type in graph.ntypes:
                if graph.num_dst_nodes(n_type) == 0:
                    continue
                output[n_type] = graph.dstnodes[n_type].data['h'].view(-1, self.num_attn_head * self.out_dim)
                if debug is True:
                    print(f'output[{n_type}] shape : {output[n_type].shape}')
        else:
            print('This is not a graph!!!!')

        if debug is True:
            print('1st gnn layer output')
            import pdb; pdb.set_trace()
            
        return output


class graphtransformer_hidden_layer_mb(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_attn_head,
                 ntype_list,
                 can_etype_list,
                 inner_type_agg_fn='mean',
                 inter_type_agg_fn='sum',
                 dropout=None,
                 device='cpu'):
        super(graphtransformer_hidden_layer_mb, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_attn_head = num_attn_head
        self.ntype_list = ntype_list
        self.can_etype_list = can_etype_list
        self.inner_type_agg_fn = inner_type_agg_fn
        self.inter_type_agg_fn = inter_type_agg_fn
        self.device = device

        # Set up the 3 linear transformation weights: query, key, and value for each edge type
        query_weight = {}
        key_weight = {}
        value_weight = {}
        for src_type, e_type, dst_type in self.can_etype_list:
            # print(src_type, e_type, dst_type)
            if query_weight.get(dst_type) is None:
                query_weight[dst_type] = nn.Linear(self.in_dim, self.out_dim * self.num_attn_head)
            if key_weight.get(src_type) is None:
                key_weight[src_type] = nn.Linear(self.in_dim, self.out_dim * self.num_attn_head)
            if value_weight.get(src_type) is None:
                value_weight[src_type] = nn.Linear(self.in_dim, self.out_dim * self.num_attn_head)

        self.query_weight = nn.ModuleDict(query_weight)
        self.key_weight = nn.ModuleDict(key_weight)
        self.value_weight = nn.ModuleDict(value_weight)
        
        # Set up attention weights
        self.node_attn_weight = nn.ModuleDict()
        for i in range(len(self.ntype_list)):
#             self.node_attn_weight[self.ntype_list[i]] = nn.Linear(self.out_dim * self.num_attn_head, 1, bias=False)
            self.node_attn_weight[self.ntype_list[i]] = nn.Linear(self.out_dim, 1, bias=False)

        # Set up droput
        if dropout is not None and dropout > 0:
            self.layer_dropout = nn.Dropout(dropout)
        else:
            self.layer_dropout = None

        # Set up layer normalization
        self.layer_norm = nn.LayerNorm(self.num_attn_head * self.out_dim)

        # Initialize all weights with uniform distribution 
        for _, linear_weight in self.query_weight.items():
            self._uniform_init_linear(linear_weight)
        for _, linear_weight in self.key_weight.items():
            self._uniform_init_linear(linear_weight)
        for _, linear_weight in self.value_weight.items():
            self._uniform_init_linear(linear_weight)
        for _, linear_weight in self.node_attn_weight.items():
            self._uniform_init_linear(linear_weight)

    def _uniform_init_linear(self, linear):
        nn.init.uniform_(linear.weight)
        if linear.bias is not None:
            nn.init.constant_(linear.bias, 0)

    def forward(self, graph, node_feats, debug=False):
        """
        Support DGL mini-batch mode.

        :param graph: an MFG in hetero mode
        :param node_feats: A dictionary including both src and dst nodes, if src and dst nodes are in the same type,
                           the first set of nodes are dst nodes.
        :return:
        """
        if graph.is_block or True:
            etype_fn = {}
            # Transform all nodes with Query, Key and Value weights
            for src_type, e_type, dst_type in graph.canonical_etypes:
                c_e_type = (src_type, e_type, dst_type)
                subgraph = graph[(src_type, e_type, dst_type)]
            
                if subgraph.num_edges() == 0:
                    continue

                # Grab
                target_nodes_TF = [node in subgraph.nodes[dst_type].data[dgl.NID] for node in graph.nodes[dst_type].data[dgl.NID]]

                if debug is True:
                    import pdb; pdb.set_trace()
                
                # Step 1: 
                query = self.query_weight[dst_type]
                key = self.key_weight[src_type]
                value = self.value_weight[src_type]
                
                # expend to 3D
                query_dst = query(node_feats[dst_type][:subgraph.dstnodes[dst_type].data[dgl.NID].shape[0]]).view(-1, self.num_attn_head, self.out_dim)
                key_src = key(node_feats[src_type]).view(-1, self.num_attn_head, self.out_dim)
                value_src = value(node_feats[src_type]).view(-1, self.num_attn_head, self.out_dim)
        
                # Step 2: compute attention values of key and query, and then put them to edges for message passing
                attn_query_dst = self.node_attn_weight[dst_type](query_dst)
                attn_key_src = self.node_attn_weight[src_type](key_src)
                
                # put to nodes data field first
                subgraph.dstnodes[dst_type].data['query'] = attn_query_dst
                subgraph.srcnodes[src_type].data['key'] = attn_key_src
                subgraph.srcnodes[src_type].data['value'] = value_src

                # Step 3: compute attention along edges. 
                subgraph.apply_edges(dglf.u_add_v('key', 'query', 'attn_head'))
                edge_heads = subgraph.edges[e_type].data['attn_head']
                edge_heads = edge_heads / (self.out_dim**0.5)

                # Step 4: do softmax. 
                edge_heads = edge_heads.view(edge_heads.shape[:-1])
                self_attn = edge_softmax(subgraph, edge_heads)

                # Step 5: compute message. 
                if self.layer_dropout is not None:
                    self_heads = self.layer_dropout(self_attn)
                else:
                    self_heads = self_attn

                self_heads = th.unsqueeze(self_heads, dim=-1)

                subgraph.edges[e_type].data['a_head'] = self_heads
                subgraph.apply_edges(lambda edges: {'attn_msg': edges.src['value'] * edges.data['a_head']})

                # Step 6: aggregation all edges into target nodes. 
                if self.inner_type_agg_fn == 'mean':
                    etype_fn[c_e_type] = (dglf.copy_e('attn_msg', 'm'), dglf.mean('m', 'h'))
                elif self.inner_type_agg_fn == 'sum':
                    etype_fn[c_e_type] = (dglf.copy_e('attn_msg', 'm'), dglf.sum('m', 'h'))
                elif self.inner_type_agg_fn == 'max':
                    etype_fn[c_e_type] = (dglf.copy_e('attn_msg', 'm'), dglf.max('m', 'h'))
                else:
                    raise KeyError('Inner type aggregator type {} not recognized.'.format(self.inner_type_agg_fn))

        # Step 7: Aggregate all different relations
        if self.inter_type_agg_fn in ['sum', 'mean', 'max']:
            graph.multi_update_all(etype_fn, self.inter_type_agg_fn)
        else:
            raise KeyError('Inter type aggregator type {} not recognized.'.format(self.inter_type_agg_fn))

        output = {}
        for n_type in graph.ntypes:
            if graph.num_dst_nodes(n_type) == 0:
                continue
            output[n_type] = graph.dstnodes[n_type].data['h'].view(-1, self.num_attn_head * self.out_dim)

        return output


class graphtransformer_output_layer(nn.Module):
    """
    This layer is an MLP, working after all GNN layers. It uses a jump loop to get the input node features directly.
    """
    def __init__(self,
                 target_emb_dim,
                 target_feat_dim,
                 hid_dim,
                 output_dim,
                 n_layer=2,
                 dropout=None,
                 target_feat_real_dim=2,
                 device='cpu'
                ):
        super(graphtransformer_output_layer, self).__init__()
        self.target_emb_dim = target_emb_dim
        self.target_feat_dim = target_feat_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.dnn_n_layer = n_layer
        self.device = device
        
        # Grab POC
        self.target_feat_real_dim = target_feat_real_dim

        # A general DNN model
        self.dnn_layers = nn.ModuleList()
        # input layer
        self.dnn_layers.append(nn.Linear(self.target_feat_dim + self.target_emb_dim, self.hid_dim))
        # hidden layers
        for i in range(self.dnn_n_layer - 2):
            self.dnn_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
        # output layer
        self.dnn_layers.append(nn.Linear(self.hid_dim, self.output_dim))

        # set up dropout
        if dropout is not None and dropout > 0:
            self.dropout = nn.Droput(dropout)
        else:
            self.dropout = None

        # set up layer norm
        self.layer_norm = nn.LayerNorm(self.hid_dim)
        
        # Grab POC
        self.target_linear = nn.Linear(self.target_feat_real_dim, self.target_feat_dim)
        
    def forward(self, target_emb, target_feat):
        """
        Support mini-batch mode. No need to change at all.

        :param target_emb:
        :param target_feat:
        :return:
        """
        
        target_feat = target_feat.type(th.FloatTensor).to(self.device)
        target_feat_resized = self.target_linear(target_feat)
        
        h = th.cat([target_emb, target_feat_resized], dim=1)

        for dnn_layer in self.dnn_layers[:-1]:
            h = dnn_layer(h)
            if self.dropout is not None:
                h = self.dropout(h)
            h = self.layer_norm(h)
            h = F.relu(h)

        output = self.dnn_layers[-1](h)

        return output

class graphtransformer_model_paper_mb(nn.Module):
    """
    This run on the N's data.
    """
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_attn_head,
                 ntype_list,
                 ntype_feature_dim_dict,
                 can_etype_list,
                 etype_feature_dim_dict,
                 num_gnn_layers,
                 target='upmid',
                 dnn_n_layer=2,
                 inner_type_agg_fn='mean',
                 inter_type_agg_fn='sum',
                 attn_merge='linear',
                 dropout=None,
#                  batch_norm=False,
                 embed_init='constant',
                 device='cpu'
                 ):
        super(graphtransformer_model_paper_mb, self).__init__()
        self.input_dim = in_dim
        self.hid_dim = hid_dim
        self.output_dim = out_dim
        self.num_attn_head = num_attn_head
        self.ntype_list = ntype_list
        self.ntype_feature_dim_dict = ntype_feature_dim_dict
        self.can_etype_list = can_etype_list
        self.etype_feature_dim_dict = etype_feature_dim_dict
        self.num_gnn_layers = num_gnn_layers
        self.target_node = target
        self.dnn_n_layer = dnn_n_layer
        self.inner_type_agg_fn = inner_type_agg_fn
        self.inter_type_agg_fn = inter_type_agg_fn
        self.dropout = dropout
#         self.batch_norm = batch_norm
        self.device = device

        # Set up initial embedding for nodes and edges, except for the target node, which is the UserID in the N's data
        # Because it is type related, so only initialize them in 1D, letting them broadcasting to nodes
        self.node_emb_dict = nn.ParameterDict({
            n_type: nn.Parameter(th.Tensor(self.input_dim).to(device)) for n_type in self.ntype_list})
        self.edge_emb_dict = nn.ParameterDict({
            e_type: nn.Parameter(th.Tensor(self.input_dim).to(device)) for (src_type, e_type, dst_type) in self.can_etype_list
        })
        
        # Initialize embedding with all 0s
        for n_type, para in self.node_emb_dict.items():
            if embed_init is 'constant':
                nn.init.constant_(para, 0.)
            elif embed_init is 'xavier_uniform':
                nn.init.xavier_uniform_(para)
            elif embed_init is 'xavier_normal':
                nn.init.xavier_normal_(para)
                
        for e_type, para in self.edge_emb_dict.items():
            if embed_init is 'constant':
                nn.init.constant_(para, 0.)
            elif embed_init is 'xavier_uniform':
                nn.init.xavier_uniform_(para)
            elif embed_init is 'xavier_normal':
                nn.init.xavier_normal_(para)

        self.gnn_layers = nn.ModuleList()
        # Set up the input layer
        self.gnn_layers.append(graphtransformer_input_layer_mb(self.input_dim,
                                                     self.hid_dim,
                                                     self.num_attn_head,
                                                     self.ntype_list,
                                                     self.ntype_feature_dim_dict,
                                                     self.can_etype_list,
                                                     self.etype_feature_dim_dict,
                                                     inner_type_agg_fn=self.inner_type_agg_fn,
                                                     inter_type_agg_fn=self.inter_type_agg_fn,
                                                     dropout=self.dropout,
                                                     device=self.device))
        # Set up the rest hidden layers
        for i in range(self.num_gnn_layers - 1):
            self.gnn_layers.append(graphtransformer_hidden_layer_mb(self.num_attn_head * self.hid_dim,
                                                          self.hid_dim,
                                                          self.num_attn_head,
                                                          self.ntype_list,
                                                          self.can_etype_list,
                                                          inner_type_agg_fn=self.inner_type_agg_fn,
                                                          inter_type_agg_fn=self.inter_type_agg_fn,
                                                          dropout=self.dropout,
                                                          device=self.device))

        # Set up the output layer
        self.output_layer = graphtransformer_output_layer(self.num_attn_head * self.hid_dim,
                                                self.input_dim,
                                                self.hid_dim,
                                                self.output_dim,
                                                device=self.device)

    def forward(self, graph, in_node_feats, out_node_feats, edge_feats_dict, perm_edge=False, debug=False):
        """
        Only target type nodes have features, all the other type nodes will use
        :param
            graph: the input semi-bipartites graph
            node_feats: features for nodes that having features
        :return:
            logits of target node.
        """
        
        def print_output_shape(graph, layer_id=0):
            print(f"subgraph for layer {layer_id} - dst numbers: {graph.dstnodes['pax'].data['_ID'].shape}")
            
        
        # input layer: process input features and embeddings to feat into the first hidden layers
        if debug is True:
            print_output_shape(graph[0], 0)
            
        h_tensors = self.gnn_layers[0](graph[0], in_node_feats, self.node_emb_dict, self.edge_emb_dict, edge_feats_dict, perm_edge)

        # GNN hidden layer
        for i, hid_layer in enumerate(self.gnn_layers[1:]):
            if debug is True:
                print_output_shape(graph[i+1], i+1)
            h_tensors = hid_layer(graph[i + 1], h_tensors)
            
        # output layer: an MLP
        # first extract target node embeddings and pass to tanh activation
        target_emb = th.tanh(h_tensors[self.target_node])

        # For mini-batch only need the output nodes' input feats
        target_feats = out_node_feats[self.target_node]
        logits = self.output_layer(target_emb, target_feats)
        
        return logits

