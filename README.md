# VOE: Vector-Based Order Embedding

Our pytorch implementation of the research work "[Order-Embeddings of Images and Language](https://arxiv.org/abs/1511.06361)".
The order relationship learning using Order Embeddings is a must baseline for pre-requiste relationship learning.
Please refer to the original [codebase](https://github.com/ivendrov/order-embeddings-wordnet) in Lua for more details.


## Dependencies
The core learning model is built using [PyTorch](https://pytorch.org/)
* Python 3.6.3
* PyTorch 0.3.0

To learn order embeddings

```
  python main.py -d <name> -h1 <dimension> -bs <batch_size> -e <#epoch> -gpu <device>
```

## Data
There should be five data files ready in the 'datasets' folder, e.g. datasets/name/
* ```<name>_split_train.pkl``` list of training instance in pickle format, each instance is a three tuple: (ent1, ent2, label), label is either -1 or 1
* ```<name>_split_dev.pkl``` list of validation instance in pickle format, instance (ent1, ent2, label) is a three tuple
* ```<name>_split_test.pkl``` list of testing instance in pickle format, instance (ent1, ent2, label) is a three tuple
* ```<name>_uid_userDoc.npy``` one hot encoding of the entities in numpy format
* ```<name>_iid_itemDoc.npy``` one hot encoding of the entities in numpy format

## Cite
Please consider cite our paper if you find the paper and the code useful.

```
@inproceedings{chiang2019one,
  title={One-class order embedding for dependency relation prediction},
  author={Chiang, Meng-Fen and Lim, Ee-Peng and Lee, Wang-Chien and Ashok, Xavier Jayaraj Siddarth and Prasetyo, Philips Kokoh},
  booktitle={Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={205--214},
  year={2019}
}
```
Feel free to send email to ankechiang@gmail.com if you have any questions. This code is modified from [ANR](https://github.com/almightyGOSU/ANR).
