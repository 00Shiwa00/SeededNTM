# Implementation of SeededNTM
This code is written based on the paper **'Enhancing Neural Topic Model with Multi-Level Supervisions from Seed Words'** by Yang Lin, Xin Gao, Xu Chu, Yasha Wang, Junfeng Zhao, and Chao Chen.

## Usage
The Repository comes with an small example that shows the necessary structures of data used:

- in the configs folder is an .yaml with the metaData used for your tests e.g. path of the Data, name of DataSet and seedWordList, Hyperparameters, number of Iterations and topics,...
- model folder contains the created embeddings and data used to train the model
- Data folder is the location where you should put your data in and finally the results of the model are saved there under Data/yourDataFolder/result

To use SeededNTM use the provided SeededNTM_Demo.ipynb as starting point for your journey and adapt it if necessary.

## References
to cite the paper of the Topic Model SeededNTM  use the following:
```
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
    pages = "13361--13377",
    abstract = "Efforts have been made to apply topic seed words to improve the topic interpretability of topic models. However, due to the semantic diversity of natural language, supervisions from seed words could be ambiguous, making it hard to be incorporated into the current neural topic models. In this paper, we propose SeededNTM, a neural topic model enhanced with supervisions from seed words on both word and document levels. We introduce a context-dependency assumption to alleviate the ambiguities with context document information, and an auto-adaptation mechanism to automatically balance between multi-level information. Moreover, an intra-sample consistency regularizer is proposed to deal with noisy supervisions via encouraging perturbation and semantic consistency. Extensive experiments on multiple datasets show that SeededNTM can derive semantically meaningful topics and outperforms the state-of-the-art seeded topic models in terms of topic quality and classification accuracy."
}
```
the Word-Level-Augmentation used and adapted from:
```
@article{xie2019unsupervised,
  title={Unsupervised Data Augmentation for Consistency Training},
  author={Xie, Qizhe and Dai, Zihang and Hovy, Eduard and Luong, Minh-Thang and Le, Quoc V},
  journal={arXiv preprint arXiv:1904.12848},
  year={2019}
}
```
the sourceCode is based on the code of keyETM, parts copied or used are marked referencing keyETM:
```
@article{harandizadeh2021keyword,
  title={Keyword Assisted Embedded Topic Model},
  author={Harandizadeh, Bahareh and Priniski, J Hunter and Morstatter, Fred},
  journal={arXiv preprint arXiv:2112.03101},
  year={2021}
}
```