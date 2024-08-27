import os
import pickle
import numpy as np
import torch
import scipy.io

### Code for the neuronal networks and functions for seededNTM (https://aclanthology.org/2023.findings-acl.845)
### inspired, copied and adapted from keyETM implementation https://www.researchgate.net/publication/356817058_Keyword_Assisted_Embedded_Topic_Model, https://github.com/bahareharandizade/KeyETM
### author: Eric Kropf (00Shiwa00)

#source: keyETM   
def _fetch(path, name):
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens.mat')
        count_file = os.path.join(path, 'bow_tr_counts.mat')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens.mat')
        count_file = os.path.join(path, 'bow_va_counts.mat')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens.mat')
        count_file = os.path.join(path, 'bow_ts_counts.mat')
    tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts = scipy.io.loadmat(count_file)['counts'].squeeze()
    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens.mat')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts.mat')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens.mat')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts.mat')
        tokens_1 = scipy.io.loadmat(token_1_file)['tokens'].squeeze()
        counts_1 = scipy.io.loadmat(count_1_file)['counts'].squeeze()
        tokens_2 = scipy.io.loadmat(token_2_file)['tokens'].squeeze()
        counts_2 = scipy.io.loadmat(count_2_file)['counts'].squeeze()
        return {'tokens': tokens, 'counts': counts,
                'tokens_1': tokens_1, 'counts_1': counts_1,
                'tokens_2': tokens_2, 'counts_2': counts_2}
    return {'tokens': tokens, 'counts': counts}

#source: keyETM   
def get_data(path):
    with open(os.path.join(path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    train = _fetch(path, 'train')
    valid = _fetch(path, 'valid')
    test = _fetch(path, 'test')

    return vocab, train, valid, test


#source: keyETM, adapted for seededNTM
def get_batch_new(tokens, counts, bow, word_augment, ind, vocab_size, device, theta_hat_d_k, emsize=300):
    """fetch input data by batch."""
    batch_size = len(ind)
    theta_hat = list()
    # batch_counts = list()
    # batch_bow = list()
    data_batch = np.zeros((batch_size, vocab_size))
    bow_batch = list()
    word_augment_batch = list()
    for i, doc_id in enumerate(ind):
        bow_batch.append(bow[doc_id])
        doc = tokens[doc_id]
        count = counts[doc_id]
        word_augment_batch.append(word_augment[doc_id])
        if len(doc) == 1:
            doc = [doc.squeeze()]
            count = [count.squeeze()]
        else:
            doc = doc.squeeze()
            count = count.squeeze()
        if doc_id != -1:
            for j, word in enumerate(doc):
                data_batch[i, word] = count[j]
    for i, doc_id in enumerate(ind):
        theta_hat.append(theta_hat_d_k[doc_id])
    theta_hat =  torch.from_numpy(np.array(theta_hat)).float().to(device)
    data_batch = torch.from_numpy(data_batch).float().to(device)
    return data_batch, bow_batch, word_augment_batch, theta_hat
