import numpy as np
import torch
import pickle

import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import matplotlib.pyplot as plt

from transformers import BertTokenizer
import re

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
MAX_LEN = 40

# Create a function to tokenize a set of texts
def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            text_pair= b[i],
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            truncation=True,
            padding='max_length',
            #pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text



def preprocessing_for_bert_single(tweet_id):
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []
    target_df = df.loc[df['tweet_id'] == tweet_id]
    sent = target_df['source'].iloc[0]
    comments = target_df['replies'].iloc[0]
    comments_str = ''
    for c in comments:
      comments_str = comments_str + c
    encoded_sent = tokenizer.encode_plus(
              text=text_preprocessing(sent),  # Preprocess sentence
              text_pair= text_preprocessing(comments_str),        # All the comments as one string
              add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
              max_length=MAX_LEN,                  # Max length to truncate/pad
              padding='max_length',         # Pad sentence to max length
              truncation=True,
              #return_tensors='pt',           # Return PyTorch tensor
              return_attention_mask=True      # Return attention mask
              )
    # Add the outputs to the lists
    input_ids.append(encoded_sent.get('input_ids'))
    attention_masks.append(encoded_sent.get('attention_mask'))
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks

def preprocessing_for_bert_combo(tweet_id):
    # Create empty lists to store outputs
    #import pandas as pd
    input_file = '/content/drive/MyDrive/Twitter_tree_2021/pair_dataframe_twitter15/twitter15_source_replies_pair.pkl'
    df = pd.read_pickle(input_file)
    input_ids = []
    attention_masks = []
    target_df = df.loc[df['tweet_id'] == tweet_id]
    sent = target_df['source'].iloc[0]
    comments = target_df['replies'].iloc[0]
    for c in comments:

      encoded_sent = tokenizer.encode_plus(
                text=text_preprocessing(sent),  # Preprocess sentence
                text_pair= c,
                add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                max_length=MAX_LEN,                  # Max length to truncate/pad
                padding='max_length',         # Pad sentence to max length
                truncation=True,
                #return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True      # Return attention mask
                )
      # Add the outputs to the lists
      input_ids.append(encoded_sent.get('input_ids'))
      attention_masks.append(encoded_sent.get('attention_mask'))
    #final_input_ids = np.array([input_ids])
    #final_attention_masks = np.array([attention_masks])
    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def preprocessing_for_bert_latest(root_node,node_content):
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    node_lst = []
    for node in node_content:
      node_lst.append(node)

    whole_lst = []
    whole_lst = root_node.tolist() + node_lst

    #print('type ', type(root_node))
    #print('root_node HERE', root_node[0])
    #print('type nodecontent 0', type(node_lst[0]))
    #print('node content 0,', node_lst[0])

    for c in whole_lst:

      encoded_sent = tokenizer.encode_plus(
                text=root_node[0],  # Preprocess sentence
                text_pair= c,
                add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                max_length=MAX_LEN,                  # Max length to truncate/pad
                padding='max_length',         # Pad sentence to max length
                truncation=True,
                #return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True      # Return attention mask
                )
      # Add the outputs to the lists
      input_ids.append(encoded_sent.get('input_ids'))
      attention_masks.append(encoded_sent.get('attention_mask'))
    #final_input_ids = np.array([input_ids])
    #final_attention_masks = np.array([attention_masks])
    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def preprocessing_for_bert_seq(root_node,node_content):
  input_ids = []
  attention_masks = []

  node_lst = []
  for node in node_content:
    node_lst.append(node)
  print('len node_lst', node_lst)
  print('type root_node ', type(root_node))
  print('len root_node ', len(root_node))
  print('rootnode[0] ', root_node[0])

  MAX_LEN_SEQ = 384
  encoded_sent = tokenizer.encode_plus(
      text = root_node[0],
      text_pair = root_node.tolist() + node_lst,
      add_special_tokens = True,
      max_length = MAX_LEN_SEQ,
      truncation = True,
      padding = 'max_length',
      return_attention_mask = True
  )
  input_ids = torch.tensor(encoded_sent.get('input_ids'))
  attention_masks = torch.tensor(encoded_sent.get('attention_mask'))

  return input_ids, attention_masks




from transformers import BertTokenizer
import re

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
MAX_LEN = 40

# Create a function to tokenize a set of texts
def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            text_pair= b[i],
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            truncation=True,
            padding='max_length',
            #pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text



def preprocessing_for_bert_single(tweet_id):
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []
    target_df = df.loc[df['tweet_id'] == tweet_id]
    sent = target_df['source'].iloc[0]
    comments = target_df['replies'].iloc[0]
    comments_str = ''
    for c in comments:
      comments_str = comments_str + c
    encoded_sent = tokenizer.encode_plus(
              text=text_preprocessing(sent),  # Preprocess sentence
              text_pair= text_preprocessing(comments_str),        # All the comments as one string
              add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
              max_length=MAX_LEN,                  # Max length to truncate/pad
              padding='max_length',         # Pad sentence to max length
              truncation=True,
              #return_tensors='pt',           # Return PyTorch tensor
              return_attention_mask=True      # Return attention mask
              )
    # Add the outputs to the lists
    input_ids.append(encoded_sent.get('input_ids'))
    attention_masks.append(encoded_sent.get('attention_mask'))
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks

def preprocessing_for_bert_combo(tweet_id):
    # Create empty lists to store outputs
    #import pandas as pd
    input_file = '/content/drive/MyDrive/Twitter_tree_2021/pair_dataframe_twitter15/twitter15_source_replies_pair.pkl'
    df = pd.read_pickle(input_file)
    input_ids = []
    attention_masks = []
    target_df = df.loc[df['tweet_id'] == tweet_id]
    sent = target_df['source'].iloc[0]
    comments = target_df['replies'].iloc[0]
    for c in comments:

      encoded_sent = tokenizer.encode_plus(
                text=text_preprocessing(sent),  # Preprocess sentence
                text_pair= c,
                add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                max_length=MAX_LEN,                  # Max length to truncate/pad
                padding='max_length',         # Pad sentence to max length
                truncation=True,
                #return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True      # Return attention mask
                )
      # Add the outputs to the lists
      input_ids.append(encoded_sent.get('input_ids'))
      attention_masks.append(encoded_sent.get('attention_mask'))
    #final_input_ids = np.array([input_ids])
    #final_attention_masks = np.array([attention_masks])
    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def preprocessing_for_bert_latest(root_node,node_content):
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    node_lst = []
    for node in node_content:
      node_lst.append(node)

    whole_lst = []
    whole_lst = root_node.tolist() + node_lst

    #print('type ', type(root_node))
    #print('root_node HERE', root_node[0])
    #print('type nodecontent 0', type(node_lst[0]))
    #print('node content 0,', node_lst[0])

    for c in whole_lst:

      encoded_sent = tokenizer.encode_plus(
                text=root_node[0],  # Preprocess sentence
                text_pair= c,
                add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                max_length=MAX_LEN,                  # Max length to truncate/pad
                padding='max_length',         # Pad sentence to max length
                truncation=True,
                #return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True      # Return attention mask
                )
      # Add the outputs to the lists
      input_ids.append(encoded_sent.get('input_ids'))
      attention_masks.append(encoded_sent.get('attention_mask'))
    #final_input_ids = np.array([input_ids])
    #final_attention_masks = np.array([attention_masks])
    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def preprocessing_for_bert_seq(root_node,node_content):
  input_ids = []
  attention_masks = []

  node_lst = []
  for node in node_content:
    node_lst.append(node)
  print('len node_lst', node_lst)
  print('type root_node ', type(root_node))
  print('len root_node ', len(root_node))
  print('rootnode[0] ', root_node[0])

  MAX_LEN_SEQ = 384
  encoded_sent = tokenizer.encode_plus(
      text = root_node[0],
      text_pair = root_node.tolist() + node_lst,
      add_special_tokens = True,
      max_length = MAX_LEN_SEQ,
      truncation = True,
      padding = 'max_length',
      return_attention_mask = True
  )
  input_ids = torch.tensor(encoded_sent.get('input_ids'))
  attention_masks = torch.tensor(encoded_sent.get('attention_mask'))

  return input_ids, attention_masks






class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default:
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accs=0
        self.F1=0
        self.F2 = 0
        self.F3 = 0
        self.F4 = 0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, acc,F1,F2,F3,F4,model,modelname,str):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.accuracy = accs
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
            self.save_checkpoint(val_loss, model,modelname,str)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("BEST Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}"
                      .format(self.accs,self.F1,self.F2,self.F3,self.F4))
        else:
            self.best_score = score
            self.accuracy = acc
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
            self.save_checkpoint(val_loss, model,modelname,str)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,modelname,str):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(),modelname+str+'.m')
        self.val_loss_min = val_loss



class EarlyStoppingBinary:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default:
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accuracy=0
        self.F1=0
        self.F2 = 0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, acc,F1,F2,model,modelname,str):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.accuracy = acc
            self.F1 = F1
            self.F2 = F2
            self.save_checkpoint(val_loss, model,modelname,str)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("BEST Accuracy: {:.4f}|T F1: {:.4f}|F F1: {:.4f}|"
                      .format(self.accs,self.F1,self.F2))
        else:
            self.best_score = score
            self.accuracy = acc
            self.F1 = F1
            self.F2 = F2
            self.save_checkpoint(val_loss, model,modelname,str)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,modelname,str):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(),modelname+str+'.m')
        self.val_loss_min = val_loss



class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def eval_gae(edges_pos, edges_neg, emb, adj_orig):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    emb = emb.data.numpy()
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []

    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []

    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])

    accuracy = accuracy_score((preds_all > 0.5).astype(float), labels_all)
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return accuracy, roc_score, ap_score


def make_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_user_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            objects.append(pickle.load(f, encoding='latin1'))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)


    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def plot_results(results, test_freq, path='results.png'):
    # Init
    plt.close('all')
    fig = plt.figure(figsize=(8, 8))

    x_axis_train = range(len(results['train_elbo']))
    x_axis_test = range(0, len(x_axis_train), test_freq)
    # Elbo
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x_axis_train, results['train_elbo'])
    ax.set_ylabel('Loss (ELBO)')
    ax.set_title('Loss (ELBO)')
    ax.legend(['Train'], loc='upper right')

    # Accuracy
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x_axis_train, results['accuracy_train'])
    ax.plot(x_axis_test, results['accuracy_test'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy')
    ax.legend(['Train', 'Test'], loc='lower right')

    # ROC
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(x_axis_train, results['roc_train'])
    ax.plot(x_axis_test, results['roc_test'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ROC AUC')
    ax.set_title('ROC AUC')
    ax.legend(['Train', 'Test'], loc='lower right')

    # Precision
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(x_axis_train, results['ap_train'])
    ax.plot(x_axis_test, results['ap_test'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('Precision')
    ax.legend(['Train', 'Test'], loc='lower right')

    # Save
    fig.tight_layout()
    fig.savefig(path)



#User features
