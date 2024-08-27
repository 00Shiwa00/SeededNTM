from multiprocessing import Pool
import scipy.special
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import time
from torch.utils.data import DataLoader
import scipy

### Code for the neuronal networks and functions for SeededNTM
### inspired and adapted from keyETM implementation https://www.researchgate.net/publication/356817058_Keyword_Assisted_Embedded_Topic_Model, https://github.com/bahareharandizade/KeyETM
### author: Eric Kropf (00Shiwa00)
class Model(nn.Module):
    """
    Code for the neuronal networks and functions for SeededNTM inspired from paper:\n
    @inproceedings{lin-etal-2023-enhancing,
    title = "Enhancing Neural Topic Model with Multi-Level Supervisions from Seed Words",
    author = "Lin, Yang  and
      Gao, Xin  and
      Chu, Xu  and
      Wang, Yasha  and
      Zhao, Junfeng  and
      Chen, Chao",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.845",
    doi = "10.18653/v1/2023.findings-acl.845",
    pages = "13361--13377"}\n
    This Code is NOT from the authors of the paper. The code may contain errors or inaccuracies. Use it at your own risk. No guarantee of correctness.

    """
    def __init__(
            self,
            device,
            theta_act,
            lambda_0,
            lambda_1,
            lambda_2,
            lambda_3,
            num_topics,
            embeddings,
            background_logword_frequencies,
            eta_s,
            w_hidden_size = 256,
            d_hidden_size = 300,
            kappa = 3,            
            enc_drop=0.2,
            debug_mode=False):
        super(Model, self).__init__()
        self.device = device
        self.theta_act = theta_act
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.kappa = kappa
        self.num_topics = num_topics
        self.d_hidden_size = d_hidden_size
        self.w_hidden_size = w_hidden_size
        
        self.embeddings = embeddings.clone().float().to(self.device)
        self.vocabulary_size, self.embedding_size = self.embeddings.size()

        #Create Embedding Layer from provided embeddings. Used for look ups
        self.embedding_layer = nn.Embedding(self.vocabulary_size, self.embedding_size)
        self.embedding_layer.from_pretrained(self.embeddings,freeze=True)
        
        
        self.m = background_logword_frequencies
        self.eta_s = eta_s
        self.enc_drop = enc_drop
        
        # Word Encoding Layers
        self.w_encoding_layer_in = nn.Linear(self.embedding_size, self.w_hidden_size)
        self.w_encoding_layer_in_ac = nn.Softplus()
        self.w_encoding_layer_hidden = nn.Linear(self.w_hidden_size, self.w_hidden_size)
        self.w_encoding_layer_hidden_ac = nn.Softplus()
        self.w_encoding_layer_out = nn.Linear(self.w_hidden_size, self.num_topics)
        self.w_encoding_layer_out_ac = nn.Softplus()
        self.w_encoding_layer_bn = nn.BatchNorm1d(self.w_hidden_size)
        self.w_encoding_layer_out_bn = nn.BatchNorm1d(self.num_topics)
        self.useBatchNorm_D = True
        # Document Encoding Layers Priors
        self.variance = 0.995
        self.prior_mean   = torch.Tensor(1, num_topics).fill_(0)
        self.prior_var    = torch.Tensor(1, num_topics).fill_(self.variance)
        self.prior_mean   = nn.Parameter(self.prior_mean, requires_grad=False)
        self.prior_var    = nn.Parameter(self.prior_var, requires_grad=False)
        self.prior_logvar = nn.Parameter(self.prior_var.log(), requires_grad=False)

        #Document Encoding Layers
        self.d_encoding_layer_in = nn.Linear(self.embedding_size, self.d_hidden_size)
        self.d_encoding_layer_in_ac = nn.Softplus()
        self.d_encoding_layer_hidden = nn.Linear(self.d_hidden_size, self.d_hidden_size)
        self.d_encoding_layer_hidden_ac = nn.Softplus()
        
        # for Reparameterization trick:
        self.mu_q_theta = nn.Linear(self.d_hidden_size, num_topics, bias=True)
        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.logsigma_q_theta = nn.Linear(self.d_hidden_size, num_topics, bias=True)
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        self.useBatchNorm_W = True
        # Parameter Vektor for deviation on m: eta_r
        #self.logeta_r = nn.Parameter(torch.Tensor((torch.rand(self.vocabulary_size, self.num_topics)-0.5)*10).to(self.device))
        self.logeta_r = nn.Parameter(torch.Tensor((torch.rand(self.num_topics, self.vocabulary_size)-0.5)*10).to(self.device))
        self.beta_bn = nn.BatchNorm1d(self.num_topics)
        self.decoder_bn = nn.BatchNorm1d(self.embedding_size)

        self.debug_mode = debug_mode    
    # Document Encoding:
    def encode(self, e_d):
        """Returns paramters of the variational distribution for \theta.

        input: e_d: tensor of shape bsz x Embedding size
        output: mu_theta, log_sigma_theta

        """
        res_in = self.d_encoding_layer_in_ac(self.d_encoding_layer_in(e_d))
        res_hidden = self.d_encoding_layer_hidden_ac(self.d_encoding_layer_hidden(res_in))
        if self.enc_drop > 0:
             q_theta =  nn.Dropout(self.enc_drop)(res_hidden)
        else:
             q_theta = res_hidden
        if self.useBatchNorm_D:
            mu_theta = self.mean_bn(self.mu_q_theta(q_theta))
            logsigma_theta = self.logvar_bn(self.logsigma_q_theta(q_theta))
        else:
            mu_theta = self.mu_q_theta(q_theta)
            logsigma_theta = self.logsigma_q_theta(q_theta)
        return mu_theta, logsigma_theta

    ## Reparameterization trick:
    def reparameterize(self, mu, logvar):
        """
        Returns a sample from a Gaussian distribution via reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul_(std).add_(mu)

    def get_theta(self, document_embeddings):
        
        """
        Computes the topic distribution for a given document embedding.

        Parameters:
        e_d (torch.Tensor): A tensor representing the document embedding of shape (bsz, embedding_size).

        Returns:
        tuple: A tuple containing three elements:
        - theta (torch.Tensor): The topic distribution for the document of shape (bsz, num_topics).
        - mu_theta (torch.Tensor): The mean of the variational distribution for theta of shape (bsz, num_topics).
        - logsigma_theta (torch.Tensor): The log variance of the variational distribution for theta of shape (bsz, num_topics).
            
            Documentation created by Tabnine
            code adapted from keyETM
        """
        mu_theta, logsigma_theta = self.encode(document_embeddings)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        return theta, mu_theta, logsigma_theta

    # Word Encoding:
    def encode_word(self, word_embedding):
        """
        Computes the word distribution for a given word embedding.

        Input:
            word_embedding (torch.Tensor): A tensor representing the word embedding of shape Vx Embeddinglength.
        Output:
            phi_nk (torch.Tensor): The topic distribution of shape (V, num_topics).
        """
        res_in = self.w_encoding_layer_in_ac(self.w_encoding_layer_in(word_embedding))
        
        if not self.useBatchNorm_W:
            res_hidden =self.w_encoding_layer_hidden_ac(self.w_encoding_layer_hidden(res_in))
        else:
            res_hidden = self.w_encoding_layer_bn(self.w_encoding_layer_hidden_ac(self.w_encoding_layer_hidden(res_in)))
        if self.enc_drop > 0:
             res_hidden =  nn.Dropout(self.enc_drop)(res_hidden)
        if self.useBatchNorm_W:
            phi_nk = self.w_encoding_layer_out_bn(self.w_encoding_layer_out_ac(self.w_encoding_layer_out(res_hidden)))
        else:
            phi_nk = self.w_encoding_layer_out_ac(self.w_encoding_layer_out(res_hidden))
        phi_nk = F.softmax(phi_nk, dim=1)

        return phi_nk
    
    def get_phi(self, word_embedding):
        return self.encode_word(word_embedding)
    
    def KL(self, a, b, returnSum = True):
        if returnSum:
            return torch.mul(a, torch.log(torch.div(a,b))).sum()
        else:
            return torch.mul(a, torch.log(torch.div(a,b))).sum(dim=1)
        
    def SKL(self, a, b):
        """symetric KL Divergence between a and b"""
        skl = self.KL(a,b) + self.KL(b,a)
        return skl
    def get_varphis_new(self, docs_theta, docs_phi_nk):
        docs_varphi = list()
        for d in range(len(docs_theta)):
            docs_varphi.append(self._get_varphi_new(docs_theta[d], docs_phi_nk[d]))
        #torch.stack(docs_varphi)
        return docs_varphi
    
    def _get_varphi_new(self, doc_theta, doc_phi_nk):   
        doc_varphi = torch.mul(doc_phi_nk, doc_theta)
        doc_varphi = torch.div(doc_varphi, torch.sum(doc_varphi, dim=1, keepdim=True))
        return doc_varphi

    def set_eta_s(self, eta_s, inverse_eta_s, seedIdxs):
        """
        seedWordIndices = list with index of each seedWord in given vocabulary.
        """
        #override existing one, default one is just a failsafe with 0 everywhere --> no seedWords
        self.eta_s = eta_s
        self.inverse_eta_s = inverse_eta_s.to(self.device)
        self.seedIdxs = seedIdxs
    def reset_eta_r(self):
        for k in range(self.num_topics):
            for id in self.seedIdxs[k]:
                self.logeta_r.data[k][id] = 0   

    def compute_L_d(self, theta, theta_hat):
        L_d = list()
        for d in range(len(theta)):
            kl_d = self.KL(theta_hat[d], theta[d])
            L_d.append(kl_d)
        L_d = torch.stack(L_d)
        return self.lambda_1 * L_d.to(self.device)

    def compute_L_w(self, phi_nk_hat, phi_nk):
        kl_list = []
        dataset = list(zip(phi_nk_hat, phi_nk))
        dataloader = DataLoader(dataset, batch_size=len(phi_nk_hat), shuffle=False)

        for batch in dataloader:
            phi_nk_hat_batch, phi_nk_batch = batch
            kl = self.KL(phi_nk_hat_batch, phi_nk_batch)
            kl_list.append(kl)

        L_w = torch.stack(kl_list)
        return self.lambda_2 * L_w.to(self.device)
    

    def compute_L_w_fast(self, phi_nk_hat, phi_nk, bows):
        
        L_w = torch.Tensor().to(self.device)
        for d in range(len(bows)):
            #get KL for each w in V
            kl_d = self.KL(phi_nk_hat[d], phi_nk, returnSum=False)
            #multiply with count of word for weighting input 
            kl_d = kl_d * bows[d]
            #remove w not in document            
            mask = bows[d]>0
            sum = torch.masked_select(kl_d, mask).sum()
            L_w = torch.cat((L_w,torch.unsqueeze(sum,0)),0)
        return L_w * self.lambda_2

    def compute_L_c(self, document_embeddings, word_level_augment, theta):
        """
        Computes the L_c terms
    
        Parameters:
        - document_embeddings (torch.Tensor): A tensor representing the document embeddings of shape (bsz, embedding_size).
        - theta (torch.Tensor): The topic distribution for the document of shape (bsz, num_topics).
    
        Returns:
        - torch.Tensor: The L_c term of shape (bsz,). The result is multiplied by lambda_3 before returning.
        """
        L_c = torch.Tensor().to(self.device)
        e_d = self.encode_word(document_embeddings)
        e_d_tick = list()
        for d in range(len(word_level_augment)):
            doc_en = self.create_en(word_level_augment[d])
            doc_e_d = self.create_e_d(doc_en)
            e_d_tick.append(doc_e_d)
        e_d_tick = torch.stack(e_d_tick)
        e_d_tick,_,_ =  self.get_theta(e_d_tick)
        L_c = list()
        for i in range(len(document_embeddings)):
            skl_1 = self.SKL(theta[i], e_d_tick[i])
            skl_2 = self.SKL(theta[i], e_d[i])
            L_c_batch = skl_1 + skl_2
            L_c.append(L_c_batch)
        L_c = torch.stack(L_c)
        return self.lambda_3 * L_c
    
    def compute_L_rec(self, varphi, beta, bows):
        L_rec = torch.Tensor().to(self.device)
        beta_log = torch.log(beta)
        for d in range(len(bows)):
            mask = torch.unsqueeze(bows[d]).expand_as(varphi[d])>0
            rec_d =torch.multiply(varphi[d], torch.transpose(beta_log,0,1)).sum(dim=1)
            sum = torch.masked_select(rec_d, mask).sum()
            L_rec = torch.cat((L_rec,torch.unsqueeze(sum,0)),0)
        return - L_rec

    def compute_L_kl_old(self, posterior_logvar, posterior_mean, varphi, theta, bows):
        posterior_var    = posterior_logvar.exp()
        prior_mean   = self.prior_mean.expand_as(posterior_mean)
        prior_var    = self.prior_var.expand_as(posterior_mean)
        prior_logvar = self.prior_logvar.expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        kld_theta = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.num_topics)

        kl = torch.Tensor().to(self.device)
        for d in range(len(varphi)):
            
            mask = bows[d]>0
            varphi_sel = torch.reshape(torch.masked_select(varphi[d], torch.unsqueeze(mask,dim=0).transpose(0,1).expand_as(varphi[d])), (-1,self.num_topics))
            theta_d_exp = theta[d].expand_as(varphi_sel).to(self.device)                 
            kl_d = self.KL(varphi_sel, theta_d_exp, returnSum=False).sum()
            
            kl = torch.cat((kl, torch.unsqueeze(kl_d,0)))
            #print(kl.size())
        kld_theta += (kl).float().to(self.device)
        kld_theta = kld_theta.sum()
        return kld_theta
    def compute_L_kl(self, posterior_logvar, posterior_mean, varphi, theta, bows):
        posterior_var = posterior_logvar.exp()
        prior_mean = self.prior_mean.expand_as(posterior_mean)
        prior_var = self.prior_var.expand_as(posterior_mean)
        prior_logvar = self.prior_logvar.expand_as(posterior_mean)
        var_division = posterior_var / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        kl_list = []

        dataset = list(zip(varphi, theta, bows))
        #print(len(dataset))
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        for batch in dataloader:
            varphi_batch, theta_batch, bow = batch
            #print(varphi_batch.size(), theta_batch.size())
            theta_batch_exp = theta_batch.expand_as(varphi_batch).to(self.device)
            kl_d = self.KL(varphi_batch, theta_batch_exp).to(self.device)
            #add numbers occurrences of words in document
            kl_d = torch.multiply(kl_d, bow)
            # remove zeros to prevent large gradient
            mask = bow>0
            kl_d = torch.masked_select(kl_d, mask)

            kl_list.append(kl_d.sum())


        L_kl = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.num_topics)
        L_kl += torch.stack(kl_list)

        return self.lambda_0 * L_kl

    
    def get_beta(self):
        beta =torch.Tensor().double().to(self.device)
        logeta_r = self.logeta_r
        for k in range(self.num_topics):
            for id in self.seedIdxs[k]:
                logeta_r.data[k][id] = 0

        beta = F.softmax(torch.add(torch.add(self.m.expand_as(logeta_r), logeta_r),self.eta_s), dim = 1)
        
        return beta
    
    # copied and adapted from keyETM code
    def decode(self, theta):
        beta = self.get_beta()
        res = torch.mm(theta, beta)
        preds = torch.log(res + 1e-6)
        return preds


    def create_en(self, wn):
        """
        input: WordIndex wn (index of word in vocabulary)
        ouput: Vektor of size self.embedding_size, representing the embedding of word wn with index wn
        """
        return self.embedding_layer(wn)

    def create_e_d(self, doc_en):
        """
        input: doc_en (tensor of size len(bow of document) x embedding_size, representing embeddings of words in document)
        ouput: e_d (tensor of size embedding_size, representing the mean of embeddings of words in document)
        """
        e_d = torch.div(doc_en.sum(dim=0),len(doc_en))
        return e_d
    def _L_rec_new(self, doc_wn, doc_varphi_nk):
        L_rec = torch.Tensor().to(self.device)
        L_rec = 0
        for n in range(len(doc_wn)):
            part = torch.sum(torch.mul(doc_varphi_nk[n],self.log_beta[doc_wn[n]]))
            L_rec += part
        L_rec = L_rec*-1
        return L_rec

    def forward(self, bows, batch_bow, word_augment_batch, theta_hat,phi_hat, theta=None):
        """
        Forward pass through the model.
        Args:
        - bows:  vector of size batchsize x vocab_size, representing bag-of-words representation of documents (counts of words)
        - batch_bow: list of Tensors of size batchsize x len(document), representing bag-of-words representation of documents using Indices of the words (include duplicates)
        - word_augment_batch: list of vector of size batchsize x list of wordIndices, representing bag-of-words representation
        - theta_hat: Document Level Supervision
        - phi_hat: Word Level Supervision
        """

        self.resetlogTime()
        #create beta
        beta = self.get_beta()
        self.log_beta = torch.log(beta.transpose(0,1))
        self.logTime("beta")

        docs_ed = list() #len = batch x embeddingsize
        docs_wn = batch_bow #len = batch x Tensor(wn) ==> contains links between embedding of word en to wn for beta_kwn
        docs_en = list() #len = batch x Tensor(wn x embeddingsize) ==> contains embedding en of word wn
        
        for d in range(len(batch_bow)): #create embeddings en for each word in document and e_d of document
            doc_en = self.create_en(batch_bow[d])
            docs_en.append(doc_en)
            e_d = self.create_e_d(doc_en)
            docs_ed.append(e_d)
        docs_ed = torch.stack(docs_ed)
        self.logTime("docs")

        #creta phi_nk for all documents using their word embeddings doc_en
        docs_phi_nk = list() # list of len batch: batch x Tensor(wn x k)
        for list_en in docs_en:
            phi_nk = self.encode_word(list_en)
            docs_phi_nk.append(phi_nk)
        self.logTime("docs_phi_nk")

        # compute theta
        if theta is None:
            theta, posterior_mean, posterior_logvar = self.get_theta(docs_ed)
        else:
            kld_theta = None
        self.logTime("theta")

        #compute varphi for each document        
        docs_varphi_nk = self.get_varphis_new(docs_phi_nk= docs_phi_nk, docs_theta=theta) #batch x Tensor(wn x k)
        self.logTime("docs_varphi_nk")

        #Compute L_rec
        starttime = time.time()
        docs_L_rec = list()
        for d in range(len(bows)):
            L_rec = torch.Tensor().to(self.device)
            selectedlogbeta = torch.index_select(self.log_beta, 0, docs_wn[d])
            L_rec = -torch.sum(torch.mul(docs_varphi_nk[d], selectedlogbeta))
            docs_L_rec.append(L_rec)
        docs_L_rec = torch.stack(docs_L_rec)
        self.logTime("L_rec")

        #compute L_kl
        #(copied and adapted from keyETM code)
        posterior_var    = posterior_logvar.exp()
        prior_mean   = self.prior_mean.expand_as(posterior_mean)
        prior_var    = self.prior_var.expand_as(posterior_mean)
        prior_logvar = self.prior_logvar.expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        kld_theta = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.num_topics)
        self.logTime("docs_L_kl1")

        docs_L_kl = list()
        for d in range(len(bows)):
            doc_varphi = docs_varphi_nk[d]  
            doc_L_kl = self.KL(doc_varphi, theta[d])
            docs_L_kl.append(doc_L_kl)
        docs_L_kl = torch.stack(docs_L_kl)
        docs_L_kl = docs_L_kl + kld_theta
        docs_L_kl = docs_L_kl * self.lambda_0
        self.logTime("docs_L_kl")

        docs_L_d = self.KL(theta, theta_hat, returnSum=False)
        docs_L_d = docs_L_d.sum()*self.lambda_1
        self.logTime("docs_L_d")
        
        docs_L_w = list()
        for d in range(len(bows)):
            selectedphi_hat = torch.index_select(phi_hat, 0, docs_wn[d])
            L_w = self.KL(selectedphi_hat,docs_phi_nk[d])
            docs_L_w.append(L_w)
        docs_L_w = torch.stack(docs_L_w)*self.lambda_2
        self.logTime("docs_L_w")
        
        docs_L_c = self.compute_L_c(docs_ed, word_augment_batch,theta)
        self.logTime("docs_L_d")

        return docs_L_rec.sum(), docs_L_kl.sum(), docs_L_d.sum(), docs_L_w.sum(), docs_L_c.sum()
    
    
    def logTime(self, logText):
        if self.debug_mode:
            if self.startTime is None:
                self.resetlogTime()
            else:
                print(f"{logText} Time taken to compute: {time.time() - self.startTime:.4f} seconds")
                self.resetlogTime()
    def resetlogTime(self):
        self.startTime = time.time()

