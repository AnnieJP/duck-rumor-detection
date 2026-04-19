import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from utils import preprocessing_for_bert_latest, preprocessing_for_bert_seq





class TextUserData(Data):
  def __init__(self, text_x, user_x, text_edge_index, user_edge_index,y,idx):
    super().__init__()
    self.text_x = text_x
    self.user_x = user_x
    self.text_edge_index = text_edge_index
    self.user_edge_index = user_edge_index
    self.y = y
    self.idx = idx

  def __inc__(self, key, value):
    if key == 'text_edge_index':
        return self.text_x.size(0)
    if key == 'user_edge_index':
        return self.user_x.size(0)
    else:
        return super().__inc__(key, value)



class DUCKData(Data):
  def __init__(self, text_x, user_x, seq_x, text_edge_index, user_edge_index,y,idx):
    super().__init__()
    self.text_x = text_x
    self.user_x = user_x
    self.seq_x = seq_x
    self.text_edge_index = text_edge_index
    self.user_edge_index = user_edge_index
    self.y = y
    self.idx = idx

  def __inc__(self, key, value):
    if key == 'text_edge_index':
        return self.text_x.size(0)
    if key == 'user_edge_index':
        return self.user_x.size(0)
    else:
        return super().__inc__(key, value)


class CommentTreeDataset(Dataset):
  def __init__(self,fold_x,data_path):
    self.fold_x = fold_x
    self.data_path = data_path

  def __len__(self):
    return len(self.fold_x)

  def __getitem__(self,index):
    id = self.fold_x[index]
    data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
    idx = int(id)
    str_idx = str(idx)
    input_ids, attention_mask = preprocessing_for_bert_latest(data['root'],data['nodecontent']) #convert list of strings to list of input_ids and attention_mask for this idx
    input_ids_seq, attention_mask_seq = preprocessing_for_bert_seq(data['root'],data['nodecontent'])

    return Data(
        edge_index = torch.LongTensor(data['edgematrix']),
        root = torch.LongTensor(data['root']),
        y = torch.LongTensor([int(data['y'])]),
        rootindex = torch.LongTensor([int(data['rootindex'])]),
        idx = torch.LongTensor([int(idx)]),
        input_ids = torch.LongTensor(input_ids),
        attention_mask = torch.LongTensor(attention_mask),
        input_ids_seq = torch.LongTensor(input_ids_seq),
        attention_mask_seq = torch.LongTensor(attention_mask_seq),
        top_index = torch.LongTensor(data['topindex']),
        tri_index = torch.LongTensor(data['triIndex'])),

def collate_fn(data):
  return data


class UserTreeDataset(Dataset):
  def __init__(self,fold_x,data_path):
    self.fold_x = fold_x
    self.data_path = data_path

  def __len__(self):
    return len(self.fold_x)

  def __getitem__(self,index):
    id = self.fold_x[index]
    data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
    idx = int(id)
    return Data(
        x = torch.tensor(data['root'],dtype=torch.float32),
        edge_index = torch.LongTensor(data['edgematrix']),
        y = torch.LongTensor([int(data['y'])]),
        idx = torch.LongTensor([int(idx)]),
    )


def collate_fn(data):
  return data




class DuckDataset(Dataset):
  def __init__(self, fold_x, data_path):
    self.fold_x = fold_x
    self.data_path = data_path

  def __len__(self):
    return len(self.fold_x)

  def __getitem__(self, index):
    id = self.fold_x[index]
    data = np.load(os.path.join(self.data_path, id + '.npz'), allow_pickle=True)
    idx = int(id)
    user_x = torch.tensor(data['userx'],dtype=torch.float32)
    user_edge_index = torch.LongTensor(data['useredgematrix'])
    text_x = torch.tensor(data['textx'],dtype=torch.float32)
    text_edge_index = torch.LongTensor(data['textedgematrix'])
    seq_x = torch.tensor(data['seqx'],dtype=torch.float32)

    y = torch.LongTensor([int(data['y'])])
    idx = torch.LongTensor([int(idx)])
    duck_data = DUCKData(text_x=text_x, user_x=user_x, seq_x=seq_x, text_edge_index=text_edge_index, user_edge_index=user_edge_index,y=y,idx=idx)
    return duck_data

def collate_fn(data):
  return data


# ---------------------------------------------------------------------------
# DUCK+ dataset — reads .npz files produced by preprocess.py
# ---------------------------------------------------------------------------

class DuckPlusDataset(Dataset):
  """
  Loads per-story .npz files written by preprocess.py.

  Each item is a torch_geometric Data object with:
    input_ids      : (N, MAX_LEN)  BERT token ids  (one row per node)
    attention_mask : (N, MAX_LEN)
    edge_index     : (2, E)        COO graph edges
    edge_feat      : (E, 6)        temporal/structural edge features
    x              : (N, 6)        user profile features (zeros if unavailable)
    y              : (1,)          label int
    rootindex      : (1,)          always 0
    topindex       : (K,)          direct children of root
    triIndex       : (T,)          nodes at depth <= 2
    idx            : (1,)          story id as int
  """

  MAX_LEN = 40

  def __init__(self, fold_x, data_path, tokenizer=None):
    self.fold_x    = fold_x
    self.data_path = data_path
    if tokenizer is None:
      from transformers import BertTokenizer
      tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                do_lower_case=True)
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.fold_x)

  def _tokenize_pair(self, src_text, reply_text):
    enc = self.tokenizer(
        text=str(src_text),
        text_pair=str(reply_text),
        add_special_tokens=True,
        max_length=self.MAX_LEN,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
    )
    return enc['input_ids'], enc['attention_mask']

  def __getitem__(self, index):
    sid  = self.fold_x[index]
    npz  = np.load(os.path.join(self.data_path, sid + '.npz'),
                   allow_pickle=True)

    root_text    = str(npz['root'][0])
    node_content = npz['nodecontent']
    edge_index   = torch.LongTensor(npz['edgematrix'])
    edge_feat    = torch.FloatTensor(npz['edge_features'])
    user_x       = torch.FloatTensor(npz['userx'])
    y            = torch.LongTensor([int(npz['y'])])
    rootindex    = torch.LongTensor([int(npz['rootindex'])])
    topindex     = torch.LongTensor(npz['topindex'])
    tri_index    = torch.LongTensor(npz['triIndex'])

    # Tokenize each node as (source, node_text) pair — DUCK's pair encoding
    all_input_ids  = []
    all_attn_masks = []
    for node_text in node_content:
      ids, mask = self._tokenize_pair(root_text, str(node_text))
      all_input_ids.append(ids)
      all_attn_masks.append(mask)

    input_ids      = torch.LongTensor(all_input_ids)
    attention_mask = torch.LongTensor(all_attn_masks)

    try:
      idx = torch.LongTensor([int(sid)])
    except ValueError:
      idx = torch.LongTensor([index])

    return Data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        edge_index=edge_index,
        edge_feat=edge_feat,
        x=user_x,
        y=y,
        rootindex=rootindex,
        top_index=topindex,
        tri_index=tri_index,
        idx=idx,
    )
