from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy import sparse
from typing import Tuple, List
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import math

### Code for the neuronal networks and functions for seededNTM (https://aclanthology.org/2023.findings-acl.845)
### inspired, copied and adapted from keyETM implementation https://www.researchgate.net/publication/356817058_Keyword_Assisted_Embedded_Topic_Model, https://github.com/bahareharandizade/KeyETM
### author: Eric Kropf (00Shiwa00)

#source: keyETM
def _remove_empty_documents(documents):
    l = list()
    #[l.append(idx) for idx,doc in  zip(range(len(documents)),documents) if doc == []]
    return [doc for doc in documents if doc != [] or len(doc) != 1], l

#source: keyETM
def _create_list_words(documents):
    return [word for document in documents for word in document]

#source: keyETM
def _create_document_indices(documents):
    aux = [[j for i in range(len(doc))] for j, doc in enumerate(documents)]
    return [int(x) for y in aux for x in y]

#source: keyETM
def _create_bow(document_indices, words, num_docs, vocab_size):
    return sparse.coo_matrix(
        ([1] *
         len(document_indices),
         (document_indices,
          words)),
        shape=(
            num_docs,
            vocab_size)).tocsr()

#source: keyETM
def _split_bow(bow_in, num_docs):
    indices = [[w for w in bow_in[doc, :].indices] for doc in range(num_docs)]
    counts = [[c for c in bow_in[doc, :].data] for doc in range(num_docs)]
    return indices, counts

#source: keyETM
def _create_dictionaries(vocabulary):
    word2id = dict([(w, j) for j, w in enumerate(vocabulary)])
    id2word = dict([(j, w) for j, w in enumerate(vocabulary)])

    return word2id, id2word

#source: keyETM
def _to_numpy_array(documents):
    return np.array([[np.array(doc) for doc in documents]],
                    dtype=object).squeeze()


def create_phi_hat(corpus, vocab, seedWordLists, tau=3):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(corpus)
    Xc = (X.T * X)

    seedWordInxList = get_seedWordIndices(vocab, seedWordLists)
    phi_hat_nk = list()
    Xc_array = Xc.toarray()
    for n_indx in range(len(vocab)):
        phi_hat = list()
        for k in range(len(seedWordInxList)):
            phi_hat_k = list()
            for seedWord_indx in seedWordInxList[k]:
                phi_hat_k.append(Xc_array[n_indx][seedWord_indx]/Xc_array[seedWord_indx][seedWord_indx])
            phi_hat.append((tau / len(seedWordInxList[k])) *np.array(phi_hat_k).sum()+1e-6)
        if np.array(phi_hat).sum() != 0:
            phi_hat_nk.append(phi_hat/np.array(phi_hat).sum())
        else:
            a = np.array([1, len(phi_hat)])
            a.fill(1)
            phi_hat_nk.append(a/len(phi_hat))
    return phi_hat_nk

def create_phi_hat_new(corpus, vocab,  seedWordLists, tau=3):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(corpus)
    Xc = (X.T * X)
    seedWordInxList = get_seedWordIndices(vocab, seedWordLists)
    phi_hat_nk = torch.Tensor()
    Xc_array = Xc.toarray()
    


    return phi_hat_nk

def get_seedWordIndices(vocab, seedWordLists):
    seedWordInxList = list()
    for seedWordList in seedWordLists:
        if seedWordList != 0:
            seedWordInx = list()

            for seedWord in seedWordList:
                finding = np.where(np.array(vocab) == seedWord)
                if len(finding[0]) != 0:
                    # print(finding)
                    seedWordInx.append(finding[0][0])
        else:
            seedWordInx = list()
            seedWordInxList.append(seedWordInx)
        seedWordInxList.append(seedWordInx)
    return seedWordInxList
def create_theta_hat(corpus, seedWordLists, vocab):   
    """
        used to create theta_hat_d_k
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(corpus)
    seedWordInxList = get_seedWordIndices(vocab, seedWordLists)
    theta_hat_d_k = list()
    
    for tfidf in X:
        tfidf_array = tfidf.toarray()
        theta_k_hat = list()
        for indx_k_list in seedWordInxList:
            tfidf_k = list()
            for indx_k in indx_k_list:
                tfidf_k.append(tfidf_array[0][indx_k])
            theta_k_hat.append(math.exp(1/len(indx_k_list)*np.array(tfidf_k).sum()))
        theta_k_hat = theta_k_hat / np.array(theta_k_hat).sum()
        theta_hat_d_k.append(theta_k_hat)
    return theta_hat_d_k

def create_mu(corpus, vocab):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform([" ".join(corpus)])
    count = X.toarray()[0]
    mu = np.log(count) - np.log(np.array(count).sum())
    return mu

def create_overall_tfidf(corpus, vocab):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform([" ".join(corpus)])
    return X.toarray()[0]

#source: keyETM and adapted for seededNTM
def create_etm_datasets(
        dataset: List[str],
        train_size=1.0,
        min_df=1,
        max_df=100.0,
        debug_mode=False) -> Tuple[list, dict, dict]:
    print(str(len(dataset)))
    """
    Creates vocabulary and train / test datasets from a given corpus. The vocabulary and datasets can
    be used to train an ETM model.

    By default, creates a train dataset with all the preprocessed documents in the corpus and an empty
    test dataset.

    This function preprocesses the given dataset, removing most and least frequent terms on the corpus - given minimum and maximum document-frequencies - and produces a BOW vocabulary.

    Parameters:
    ===
        dataset (list of str): original corpus to be preprocessed. Is composed by a list of sentences
        train_size (float): fraction of the original corpus to be used for the train dataset. By default, uses entire corpus
        min_df (float): Minimum document-frequency for terms. Removes terms with a frequency below this threshold
        max_df (float): Maximum document-frequency for terms. Removes terms with a frequency above this threshold
        debug_mode (bool): Wheter or not to log function's operations to the console. By default, no logs are made

    Returns:
    ===
        vocabulary (list of str): words vocabulary. Doesn't includes words not in the training dataset
        train_dataset (dict): BOW training dataset, split in tokens and counts. Must be used on ETM's fit() method.
        test_dataset (dict): BOW testing dataset, split in tokens and counts. Can be use on ETM's perplexity() method.
    """
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
    vectorized_documents = vectorizer.fit_transform(dataset)

    documents_without_stop_words = [
        [word for word in document.split()
            if word not in vectorizer.stop_words_]
        for document in dataset]

    signed_documents = vectorized_documents.sign()

    if debug_mode:
        print('Building vocabulary...')

    sum_counts = signed_documents.sum(axis=0)
    v_size = sum_counts.shape[1]
    sum_counts_np = np.zeros(v_size, dtype=int)
    for v in range(v_size):
        sum_counts_np[v] = sum_counts[0, v]
    word2id = dict([(w, vectorizer.vocabulary_.get(w))
                    for w in vectorizer.vocabulary_])
    id2word = dict([(vectorizer.vocabulary_.get(w), w)
                    for w in vectorizer.vocabulary_])

    if debug_mode:
        print('Initial vocabulary size: {}'.format(v_size))

    # Sort elements in vocabulary
    idx_sort = np.argsort(sum_counts_np)
    # Creates vocabulary
    vocabulary = [id2word[idx_sort[cc]] for cc in range(v_size)]

    if debug_mode:
        print('Tokenizing documents and splitting into train/test...')

    num_docs = signed_documents.shape[0]
    train_dataset_size = int(np.floor(train_size * num_docs))
    test_dataset_size = int(num_docs - train_dataset_size)
    #idx_permute = np.random.permutation(num_docs).astype(int)

    # Remove words not in train_data
    #vocabulary = list(set([w for idx_d in range(train_dataset_size)
    #                       for w in documents_without_stop_words[idx_permute[idx_d]] if w in word2id]))
    vocabulary = list(set([w for idx_d in range(train_dataset_size)
                           for w in documents_without_stop_words[idx_d] if w in word2id]))
    # Create dictionary and inverse dictionary
    word2id, id2word = _create_dictionaries(vocabulary)

    if debug_mode:
        print(
            'vocabulary after removing words not in train: {}'.format(
                len(vocabulary)))

    #docs_train = [[word2id[w] for w in documents_without_stop_words[idx_permute[idx_d]]
    #               if w in word2id] for idx_d in range(train_dataset_size)]
    docs_train = [[word2id[w] for w in documents_without_stop_words[idx_d]
                   if w in word2id] for idx_d in range(train_dataset_size)]
    #docs_test = [
    #    [word2id[w] for w in
    #        documents_without_stop_words[idx_permute[idx_d + train_dataset_size]]
    #     if w in word2id] for idx_d in range(test_dataset_size)]
    docs_test = [
        [word2id[w] for w in
            documents_without_stop_words[idx_d + train_dataset_size]
         if w in word2id] for idx_d in range(test_dataset_size)]
    if debug_mode:
        print(
            'Number of documents (train_dataset): {} [this should be equal to {}]'.format(
                len(docs_train),
                train_dataset_size))
        print(
            'Number of documents (test_dataset): {} [this should be equal to {}]'.format(
                len(docs_test),
                test_dataset_size))

    if debug_mode:
        print('Removing empty documents...')

    #docs_train, removed_idx_train = _remove_empty_documents(docs_train)
    docs_test, removed_idx_test = _remove_empty_documents(docs_test)

    # Remove test documents with length=1
    docs_test = [doc for doc in docs_test if len(doc) > 1]

    # Obtains the training and test datasets as word lists
    words_train = [[id2word[w] for w in doc] for doc in docs_train]
    wordset_train = words_train
    words_test = [[id2word[w] for w in doc] for doc in docs_test]

    docs_test_h1 = [[w for i, w in enumerate(
        doc) if i <= len(doc) / 2.0 - 1] for doc in docs_test]
    docs_test_h2 = [[w for i, w in enumerate(
        doc) if i > len(doc) / 2.0 - 1] for doc in docs_test]

    words_train = _create_list_words(docs_train)
    words_test = _create_list_words(docs_test)
    words_ts_h1 = _create_list_words(docs_test_h1)
    words_ts_h2 = _create_list_words(docs_test_h2)

    if debug_mode:
        print('len(words_train): ', len(words_train))
        print('len(words_test): ', len(words_test))
        print('len(words_ts_h1): ', len(words_ts_h1))
        print('len(words_ts_h2): ', len(words_ts_h2))

    doc_indices_train = _create_document_indices(docs_train)
    doc_indices_test = _create_document_indices(docs_test)
    doc_indices_test_h1 = _create_document_indices(docs_test_h1)
    doc_indices_test_h2 = _create_document_indices(docs_test_h2)

    if debug_mode:
        print('len(np.unique(doc_indices_train)): {} [this should be {}]'.format(
            len(np.unique(doc_indices_train)), len(docs_train)))
        print('len(np.unique(doc_indices_test)): {} [this should be {}]'.format(
            len(np.unique(doc_indices_test)), len(docs_test)))
        print('len(np.unique(doc_indices_test_h1)): {} [this should be {}]'.format(
            len(np.unique(doc_indices_test_h1)), len(docs_test_h1)))
        print('len(np.unique(doc_indices_test_h2)): {} [this should be {}]'.format(
            len(np.unique(doc_indices_test_h2)), len(docs_test_h2)))

    # Number of documents in each set
    n_docs_train = len(docs_train)
    n_docs_test = len(docs_test)
    n_docs_test_h1 = len(docs_test_h1)
    n_docs_test_h2 = len(docs_test_h2)

    bow_train = _create_bow(
        doc_indices_train,
        words_train,
        n_docs_train,
        len(vocabulary))
    bow_test = _create_bow(
        doc_indices_test,
        words_test,
        n_docs_test,
        len(vocabulary))
    bow_test_h1 = _create_bow(
        doc_indices_test_h1,
        words_ts_h1,
        n_docs_test_h1,
        len(vocabulary))
    bow_test_h2 = _create_bow(
        doc_indices_test_h2,
        words_ts_h2,
        n_docs_test_h2,
        len(vocabulary))

    bow_train_tokens, bow_train_counts = _split_bow(bow_train, n_docs_train)
    bow_test_tokens, bow_test_counts = _split_bow(bow_test, n_docs_test)
    bow_test_h1_tokens, bow_test_h1_counts = _split_bow(
        bow_test_h1, n_docs_test_h1)
    bow_test_h2_tokens, bow_test_h2_counts = _split_bow(
        bow_test_h2, n_docs_test_h2)

    train_dataset = {
        'tokens': _to_numpy_array(bow_train_tokens),
        'counts': _to_numpy_array(bow_train_counts),
        'bow'   : docs_train,
        'words' : wordset_train
    }

    test_dataset = {
        'test': {
            'tokens': _to_numpy_array(bow_test_tokens),
            'counts': _to_numpy_array(bow_test_counts),
        },
        'test1': {
            'tokens': _to_numpy_array(bow_test_h1_tokens),
            'counts': _to_numpy_array(bow_test_h1_counts),
        },
        'test2': {
            'tokens': _to_numpy_array(bow_test_h2_tokens),
            'counts': _to_numpy_array(bow_test_h2_counts),
        }
    }
    # print(str(len(bow_train_tokens)))
    return vocabulary, train_dataset, test_dataset

#source: keyETM
def initialize_embeddings(vocab,vectors):
       

        #if use_c_format_w2vec:
           # vectors = self._get_embeddings_from_original_word2vec(embeddings)
        #elif isinstance(embeddings, str):
            #if self.debug_mode:
                #print('Reading embeddings from word2vec file...')
            #vectors = KeyedVectors.load(embeddings, mmap='r')

        model_embeddings = np.zeros((len(vocab), 300))

        for i, word in enumerate(vocab):
            try:
                model_embeddings[i] = vectors[word]
            except KeyError:
                model_embeddings[i] = np.random.normal(
                    scale=0.6, size=(self.emb_size, ))
        return model_embeddings


#source:keyETM
def nearest_neighbors_m(query, vectors,thre):
    #vectors = limit_embed.vectors
    #index = vocab.index(word)
    #query = vectors[index]
    print(vectors.shape)
    ranks = vectors.dot(query).squeeze()
    print(ranks)
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    print(ranks)
    return [(r,i) for r,i in zip(ranks,range(len(ranks))) if r>=thre]


#source:keyETM
def get_gamma_prior(vocab,seedwords,n_latent,bs,embeddings):
    limit_embed = initialize_embeddings(vocab,embeddings)
    gamma_prior = np.zeros((len(vocab),n_latent))
    gamma_prior_bin = np.zeros((bs,len(vocab), n_latent))
    for idx_topic, seed_topic in enumerate(seedwords):
        if len(seed_topic) != 0:
            topic_vect = []
            for idx_word, seed_word in enumerate(seed_topic):
                if seed_word in vocab:
                    idx_vocab = vocab.index(seed_word)

                    #print(seed_word)
                    #print(idx_vocab)
                    gamma_prior[idx_vocab, idx_topic] = 1.0 
                    gamma_prior_bin[:, idx_vocab, :]=1.0
                    topic_vect.append(embeddings[seed_word])
                else:
                    print(seed_word)
            tv = sum(topic_vect)/len(topic_vect)
            rank = nearest_neighbors_m(tv, limit_embed,0.5)
            #print(rank)
            for item in rank:
                #print(vocab[item[1]])
                if(gamma_prior[item[1], idx_topic]!=1.0):
                     #gamma_prior[item[1], idx_topic]=round(item[0],2)
                     gamma_prior[item[1], idx_topic]=1.0
                     gamma_prior_bin[:, item[1], :]= 1.0
    #print(gamma_prior[:250])
    return torch.from_numpy(gamma_prior),torch.from_numpy(gamma_prior_bin)
#source:keyETM
def  read_seedword(seedword_path):
    with open(seedword_path, 'r') as f:
        return [l.replace('\n','').split(',') for l in f]