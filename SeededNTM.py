from __future__ import print_function

import os
import pandas as pd
import numpy as np
import math
import torch
from torch import optim
from typing import List
from gensim.models import Word2Vec, KeyedVectors
from model import Model
from utils import data, embedding, metrics
from torch import nn
from tqdm import tqdm
import time
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt 


### Code for the neuronal networks and functions for seededNTM (https://aclanthology.org/2023.findings-acl.845)
### inspired, copied and adapted from keyETM implementation https://www.researchgate.net/publication/356817058_Keyword_Assisted_Embedded_Topic_Model, https://github.com/bahareharandizade/KeyETM
### author: Eric Kropf (00Shiwa00)

class SNTM(object):
    """
    Creates an embedded topic model instance. The model hyperparameters are:

        vocabulary (list of str): training dataset vocabulary
        embeddings (str or KeyedVectors): KeyedVectors instance containing word-vector mapping for embeddings, or its path
        use_c_format_w2vec (bool): wheter input embeddings use word2vec C format. Both BIN and TXT formats are supported
        model_path (str): path to save trained model. If None, the model won't be automatically saved
        batch_size (int): input batch size for training
        num_topics (int): number of topics
        lambda_0 (float) = 0.01 as KL annealing factor
        lambda_1 (float) = 10.0 set strength of Document Level Supervision
        lambda_2 (float) = 10.0 set strength of Word Level Supervision
        lambda_3 (float) = 10.0 set strength of Noise-Reduction Consistency Regularizer
        kappa (float) = seeding strength
        rho_size (int): dimension of rho
        emb_size (int): dimension of embeddings
        t_hidden_size (int): dimension of hidden space of q(theta)
        theta_act (str): tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)
        lr (float): learning rate
        lr_factor (float): divide learning rate by this...
        epochs (int): number of epochs to train. 150 for 20ng 100 for others
        optimizer_type (str): choice of optimizer
        seed (int): random seed (default: 2024)
        enc_drop (float): dropout rate on encoder
        clip (float): gradient clipping
        nonmono (int): number of bad hits allowed
        wdecay (float): some l2 regularization
        anneal_lr (bool): whether to anneal the learning rate or not
        num_words (int): number of words for topic viz
        log_interval (int): when to log training
        visualize_every (int): when to visualize results
        eval_batch_size (int): input batch size for evaluation
        eval_perplexity (bool): whether to compute perplexity on document completion task
        debug_mode (bool): wheter or not should log model operations
    """

    def __init__(
        self,
        vocabulary,
        embeddings=None,
        use_c_format_w2vec=False,
        model_path=None,
        batch_size=1000,
        num_topics=50,
        rho_size=300,
        emb_size=300,
        t_hidden_size=300,
        theta_act='relu',
        train_embeddings=False,
        lr=0.005,
        lambda_0 = 0.01,
        lambda_1=10.0,
        lambda_2=10.0,
        lambda_3=10.0,
        kappa = 3,
        lr_factor=4.0,
        epochs=20,
        #optimizer_type='adam',
        seed=2024,
        enc_drop=0.2,
        clip=0.0,
        nonmono=10,
        wdecay=1.2e-6,
        anneal_lr=False,
        num_words=10,
        log_interval=2,
        visualize_every=10,
        eval_batch_size=1000,
        eval_perplexity=False,
        debug_mode = True,
        m = None,
        eta_s= None,
        visualize_doc_topic_dist = True
    ):
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_topics = num_topics
        self.rho_size = rho_size
        self.emb_size = emb_size
        self.t_hidden_size = t_hidden_size
        self.theta_act = theta_act
        self.lr_factor = lr_factor
        self.epochs = epochs
        self.seed = seed
        self.enc_drop = enc_drop
        self.clip = clip
        self.nonmono = nonmono
        self.anneal_lr = anneal_lr
        self.debug_mode = debug_mode
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.kappa = kappa
        self.m = m
        self.eta_s = eta_s
        self.num_words = num_words
        self.log_interval = log_interval
        self.visualize_every = visualize_every
        self.visualize_doc_topic_dist = visualize_doc_topic_dist
        self.eval_batch_size = eval_batch_size
        self.eval_perplexity = eval_perplexity
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        self.embeddings = None if train_embeddings else self.initialize_embeddings(
            embeddings, use_c_format_w2vec=use_c_format_w2vec)
        self.model = Model(
            self.device,
            self.theta_act,            
            self.lambda_0,
            self.lambda_1,
            self.lambda_2,
            self.lambda_3,
            self.num_topics,
            self.embeddings,
            self.m,
            self.eta_s,
            d_hidden_size = self.t_hidden_size,
            kappa = self.kappa,
            enc_drop = self.enc_drop
            ).to(
            self.device)
        self.optimizer = self._get_optimizer("adam", lr, wdecay)
        print(self.optimizer)
    def __str__(self):
        return f'{self.model}'

    #source: keyETM
    def _get_extension(self, path):
        assert isinstance(path, str), 'path extension is not str'
        filename = path.split(os.path.sep)[-1]
        return filename.split('.')[-1]
        
    #source: keyETM    
    def initialize_embeddings(
    self, 
    embeddings,
    use_c_format_w2vec=False
    ):
        vectors = embeddings if isinstance(embeddings, KeyedVectors) else {}

        if use_c_format_w2vec:
            vectors = self._get_embeddings_from_original_word2vec(embeddings)
        elif isinstance(embeddings, str):
            if self.debug_mode:
                print('Reading embeddings from word2vec file...')
            vectors = KeyedVectors.load(embeddings, mmap='r')

        model_embeddings = np.zeros((self.vocabulary_size, self.emb_size))

        for i, word in enumerate(self.vocabulary):
            try:
                model_embeddings[i] = vectors[word]
            except KeyError:
                model_embeddings[i] = np.random.normal(
                    scale=0.6, size=(self.emb_size, ))
        return torch.from_numpy(model_embeddings).to(self.device)
    #source: keyETM
    def _get_embeddings_from_original_word2vec(self, embeddings_file):
        if self._get_extension(embeddings_file) == 'txt':
            if self.debug_mode:
                print('Reading embeddings from original word2vec TXT file...')
            vectors = {}
            iterator = embedding.MemoryFriendlyFileIterator(embeddings_file)
            for line in iterator:
                word = line[0]
                if word in self.vocabulary:
                    vect = np.array(line[1:]).astype(np.float)
                    vectors[word] = vect
            return vectors
        elif self._get_extension(embeddings_file) == 'bin':
            if self.debug_mode:
                print('Reading embeddings from original word2vec BIN file...')
            return KeyedVectors.load_word2vec_format(
                embeddings_file, 
                binary=True
            )
        else:
            raise Exception('Original Word2Vec file without BIN/TXT extension')

    #source: keyETM
    def _get_optimizer(self, optimizer_type, learning_rate, wdecay):
        paramterlist = list()
        eta_r = torch.Tensor()
        first = True
        for key in self.model.parameters():
            if key.requires_grad:
                if first:
                    first = False
                    eta_r = key
                else:
                    paramterlist.append(key)

        if optimizer_type == 'adam':
            opt = optim.Adam(paramterlist,lr=learning_rate,weight_decay=wdecay)
            opt.add_param_group({'params': eta_r, 'lr': 0.001, 'weight_decay': wdecay})
            return opt
        elif optimizer_type == 'adagrad':
            return optim.Adagrad(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=wdecay)
        elif optimizer_type == 'adadelta':
            return optim.Adadelta(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=wdecay)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=wdecay)
        elif optimizer_type == 'asgd':
            return optim.ASGD(
                self.model.parameters(),
                lr=learning_rate,
                t0=0,
                lambd=0.,
                weight_decay=wdecay)
        elif optimizer_type == 'sgd':
            return optim.SGD(
                # paramterlist,
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=wdecay)
        else:
            if self.debug_mode:
                print('Defaulting to vanilla SGD')
            return optim.SGD(self.model.parameters(), lr=learning_rate)
    #source: keyETM and adapted for sNTM
    def _set_training_data(self, train_data, theta_hat_d_k,  phi_hat_n_k, backgroundWordFrequencies, seedWordIndices):
        self.train_tokens = train_data['tokens']
        self.train_counts =  train_data['counts']
        self.bow = train_data['bow']
        if 'word_level_augment' in train_data:
            wla = list()
            #wenn wortlisten übergeben werden: diese zuerst in Embeddingsvektoren überführen
            for word_list in train_data['word_level_augment']:
                embeddingvectorlist = list()
                for word in word_list:
                    embeddingvectorlist.append(self.vocabulary.index(word))
                wla_part = torch.from_numpy(np.array(embeddingvectorlist)).to(self.device)
                wla.append(wla_part)
            self.word_level_aug = wla
        
        self.theta_hat_d_k = theta_hat_d_k
        self.num_docs_train = len(self.train_tokens)
        self.phi_hat_n_k = phi_hat_n_k
        self.phi_hat_n_k =  torch.from_numpy(np.array(phi_hat_n_k)).float().to(self.device)
        self.model.m = torch.from_numpy(np.array(backgroundWordFrequencies)).float().to(self.device)
        
        self.seedWordIndices = list()
        for slist in seedWordIndices:
            self.seedWordIndices.append(torch.from_numpy(np.array(slist)).to(self.device))
        eta_s = torch.zeros(self.num_topics, self.vocabulary_size).to(self.device)
        inverse_eta_s = torch.ones(self.num_topics, self.vocabulary_size).to(self.device)
        for k in range(self.num_topics):
            for idx in seedWordIndices[k]:
                eta_s[k][idx] = self.kappa
                inverse_eta_s[k][idx] = 0
        self.model.set_eta_s(eta_s, inverse_eta_s, seedWordIndices)
        print("fin set data")
    #source: keyETM and adapted for sNTM
    def fit(self, train_data, theta_hat_d_k, phi_hat_n_k, backgroundWordFrequencies, seedWordIndices,CategoryIds=None, test_data=None):
        print("fit")
        self._set_training_data(train_data, theta_hat_d_k, phi_hat_n_k, backgroundWordFrequencies, seedWordIndices)
        self.categoryIds = CategoryIds
        print("set data")
        self.model.to(self.model.device)
        best_val_ppl = 1e9
        all_val_ppls = []
        print(str(self.epochs))
        for epoch in range(1, self.epochs):
            print("train epoch: "+str(epoch))
            self._trainSNTM(epoch)
            if self.eval_perplexity:
                val_ppl = self._perplexity(
                    test_data)
                if val_ppl < best_val_ppl:
                    if self.model_path is not None:
                        self._save_model(self.model_path)
                    best_val_ppl = val_ppl
                else:
                    # check whether to anneal lr
                    lr = self.optimizer.param_groups[0]['lr']
                    if self.anneal_lr and (len(all_val_ppls) > self.nonmono and val_ppl > min(
                            all_val_ppls[:-self.nonmono]) and lr > 1e-5):
                        self.optimizer.param_groups[0]['lr'] /= self.lr_factor

                all_val_ppls.append(val_ppl)

            if self.debug_mode:
                for row in self.get_topics():
                    print(*row, sep="\t")
            if epoch % self.visualize_every == 0:
                self.visualize_documents()
        if self.model_path is not None:
            self._save_model(self.model_path)

        if self.eval_perplexity and self.model_path is not None:
            self._load_model(self.model_path)
            val_ppl = self._perplexity(train_data)

        return self
    #source: keyETM and adapted for sNTM
    def _trainSNTM(self, epoch):
        torch.autograd.set_detect_anomaly(True)
        self.model.train()
        acc_loss = 0
        acc_kl_theta_loss = 0
        cnt = 0
        acc_gl_loss = 0 
        acc_L_d_loss = 0 
        acc_L_w_loss = 0 
        acc_L_c_loss = 0 
        #get batch of Documents:
        indices = torch.randperm(self.num_docs_train)
        indices = torch.split(indices, self.batch_size)
        self.model.lambda_0 = min(self.model.lambda_0 + 1/self.epochs, 1.0)
        self.model.reset_eta_r()
        for idx, ind in tqdm(enumerate(indices), desc=f'batch {epoch}', total=len(indices)):
            self.optimizer.zero_grad()
            self.model.zero_grad()
            self.model.reset_eta_r()
            data_batch, bow_batch, word_augment_batch, theta_hat = data.get_batch_new(
                self.train_tokens,
                self.train_counts,
                self.bow,
                self.word_level_aug,
                ind,
                self.vocabulary_size,
                self.device,
                self.theta_hat_d_k
                )
            recon_loss, kld_theta,L_d,L_w,L_c = self.model(
                data_batch, bow_batch, word_augment_batch, theta_hat, self.phi_hat_n_k)
            
            total_loss = recon_loss + kld_theta + L_d + L_w + L_c
            total_loss.backward()
            if self.clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip)
            self.optimizer.step()
            acc_gl_loss += torch.sum(L_d + L_w + L_c ).item()
            acc_L_d_loss += L_d.item()
            acc_L_w_loss += L_w.item()
            acc_L_c_loss += L_c.item()
            acc_loss += torch.sum(recon_loss).item()
            acc_kl_theta_loss += torch.sum(kld_theta).item()
            cnt += 1
            
            if idx % self.log_interval == 0 and idx > 0:
                cur_loss = round(acc_loss / cnt, 2)
                cur_GL = round(acc_gl_loss / cnt, 2)
                cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
                cur_real_loss = round(cur_loss + cur_kl_theta+cur_GL , 2)

        cur_loss = round(acc_loss / cnt, 2)
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
        cur_GL = round(acc_gl_loss / cnt, 2)
        cur_real_loss = round(cur_loss + cur_kl_theta+cur_GL, 2)
        
        if self.debug_mode:
            print('Epoch {} - Learning Rate: {} - KL theta: {} - Rec loss: {} - PLoss: {} - NELBO: {} - L_d:{} - L_w: {} - L_c:{} -'.format(
                epoch, self.optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_GL,cur_real_loss,acc_L_d_loss,acc_L_w_loss,acc_L_c_loss))    
            
    #source: keyETM
    def get_topic_word_matrix(self) -> List[List[str]]:
        """
        Obtains the topic-word matrix learned for the model.

        The topic-word matrix lists all words for each discovered topic.
        As such, this method will return a matrix representing the words.

        Returns:
        ===
            list of list of str: topic-word matrix.
            Example:
                [['world', 'planet', 'stars', 'moon', 'astrophysics'], ...]
        """
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            beta = self.model.get_beta()

            topics = []

            for i in range(self.num_topics):
                words = list(beta[i].cpu().numpy())
                topic_words = [self.vocabulary[a] for a, _ in enumerate(words)]
                topics.append(topic_words)

            return topics
    #source: keyETM
    def get_topic_word_dist(self) -> torch.Tensor:
        """
        Obtains the topic-word distribution matrix.

        The topic-word distribution matrix lists the probabilities for each word on each topic.

        This is a normalized distribution matrix, and as such, each row sums to one.

        Returns:
        ===
            torch.Tensor: topic-word distribution matrix, with KxV dimension, where
            K is the number of topics and V is the vocabulary size
            Example:
                tensor([[3.2238e-04, 3.7851e-03, 3.2811e-04, ..., 8.4206e-05, 7.9504e-05,
                4.0738e-04],
                [3.6089e-05, 3.0677e-03, 1.3650e-04, ..., 4.5665e-05, 1.3241e-04,
                5.8661e-05]])
        """
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            return self.model.get_beta()
    #source: keyETM and adapted for sNTM
    def get_document_topic_dist(self) -> torch.Tensor:
        """
        Obtains the document-topic distribution matrix.

        The document-topic distribution matrix lists the probabilities for each topic on each document.

        This is a normalized distribution matrix, and as such, each row sums to one.

        Returns:
        ===
            torch.Tensor: topic-word distribution matrix, with DxK dimension, where
            D is the number of documents in the corpus and K is the number of topics
            Example:
                tensor([[0.1840, 0.0489, 0.1020, 0.0726, 0.1952, 0.1042, 0.1275, 0.1657],
                [0.1417, 0.0918, 0.2263, 0.0840, 0.0900, 0.1635, 0.1209, 0.0817]])
        """
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            indices = torch.tensor(range(self.num_docs_train))
            indices = torch.split(indices, self.batch_size)

            thetas = []



            for idx, ind in enumerate(indices):
                data_batch, bow_batch,_, theta_hat = data.get_batch_new(
                # data_batch, batch_counts, batch_bow, theta_hat = data.get_batch_new(
                    self.train_tokens,
                    self.train_counts,
                    self.bow,
                    self.word_level_aug,
                    ind,
                    self.vocabulary_size,
                    self.device,
                    self.theta_hat_d_k
                    )
                
                docs_ed = list() #len = batch x embeddingsize
                docs_wn = bow_batch #len = batch x Tensor(wn) ==> contains links between embedding of word en to wn for beta_kwn
                docs_en = list() #len = batch x Tensor(wn x embeddingsize) ==> contains embedding en of word wn
                
                for d in range(len(bow_batch)):
                    doc_en = self.model.create_en(bow_batch[d])
                    if torch.isnan(doc_en).any():
                        print(bow_batch[d], doc_en)
                    docs_en.append(doc_en)
                    e_d = self.model.create_e_d(doc_en)
                    docs_ed.append(e_d)
                docs_ed = torch.stack(docs_ed)
                theta, _, _ = self.model.get_theta(docs_ed)


                thetas.append(theta)
            return torch.cat(tuple(thetas), 0)
    #source: keyETM and adapted for sNTM
    def get_topic_coherence(self, top_n=10) -> float:
        """
        Calculates NPMI topic coherence for the model.

        By default, considers the 10 most relevant terms for each topic in coherence computation.

        Parameters:
        ===
            top_n (int): number of words per topic to consider in coherence computation

        Returns:
        ===
            float: the model's topic coherence
        """
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            #beta = self.model.get_beta().data.cpu().numpy()
            beta = self.model.get_beta()
            return metrics.get_topic_coherence(
                beta, self.train_tokens, self.vocabulary, top_n)
    #source: keyETM and adapted for sNTM
    def get_topic_diversity(self, top_n=25) -> float:
        """
        Calculates topic diversity for the model.

        By default, considers the 25 most relevant terms for each topic in diversity computation.

        Parameters:
        ===
            top_n (int): number of words per topic to consider in diversity computation

        Returns:
        ===
            float: the model's topic diversity
        """
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            #beta = self.model.get_beta().data.cpu().numpy()
            beta = self.model.get_beta()
            return metrics.get_topic_diversity(beta, top_n)
    #source: keyETM
    def _save_model(self, model_path):
        assert self.model is not None, \
            'no model to save'

        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, 'wb') as file:
            torch.save(self.model, file)
    #source: keyETM
    def _load_model(self, model_path):
        assert os.path.exists(model_path), \
            "model path doesn't exists"

        with open(model_path, 'rb') as file:
            self.model = torch.load(file)
            self.model = self.model.to(self.device)
    #source: keyETM and adapted for sNTM
    def get_topics(self, top_n_words=10) -> List[str]:
        """
        Gets topics. By default, returns the 10 most relevant terms for each topic.

        Parameters:
        ===
            top_n_words (int): number of top words per topic to return

        Returns:
        ===
            list of str: topic list
        """

        with torch.no_grad():
            topics = []
            gammas = self.model.get_beta()

            for k in range(self.num_topics):
                gamma = gammas[k]
                top_words = list(gamma.cpu().numpy().argsort()
                                 [-top_n_words:][::-1])
                topic_words = [self.vocabulary[a] for a in top_words]
                topics.append(topic_words)

            return topics
    
    def visualize_documents(self):
       
        if self.visualize_doc_topic_dist:
            if self.categoryIds is not None:
                print("Visualizing ...")
                dist = self.get_document_topic_dist()
                df_analyse = pd.DataFrame()
                #df_t = df_t[df_t["CategoryId"]<5]
                df_analyse["true_label"] = self.categoryIds
                # print(df_analyse)
                distlist = list()
                # print(dist.size())
                dist = dist.cpu()
                for d in dist:
                    distlist.append(d.argmax().item())
                df_analyse["predicted_label"] = distlist
                # print(df_analyse)
                l = np.zeros((self.num_topics,self.num_topics), dtype=int)
                for i in range(self.num_topics):
                    d = df_analyse[df_analyse["true_label"] == i]
                    vc = d['predicted_label'].value_counts()
                    for index, value in vc.items():
                        l[i][int(index)] = value
                # print(l)
                
                print(classification_report(self.categoryIds, distlist))
                plt.show(sns.heatmap(l))
                
                print("Done.")
                