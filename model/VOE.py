import torch
import torch.nn as nn
import torch.nn.functional as F

from .utilities import PAD_idx, UNK_idx

from .ANR_ARL import ANR_ARL

from .ANRS_RatingPred import ANRS_RatingPred

from tqdm import tqdm



class VOE(nn.Module):

	def __init__(self, logger, args, num_users, num_items):
	# def __init__(self, logger, args, num_users):

		super(VOE, self).__init__()

		self.logger = logger
		self.args = args

		self.num_users = num_users
		self.num_items = num_items


		# User Documents & Item Documents (Input)
		self.uid_userDoc = nn.Embedding(self.num_users, self.args.max_doc_len)
		self.uid_userDoc.weight.requires_grad = True

		self.iid_itemDoc = nn.Embedding(self.num_items, self.args.max_doc_len)
		self.iid_itemDoc.weight.requires_grad = True

		# Word Embeddings (Input)
		self.wid_wEmbed = nn.Embedding(self.args.vocab_size, self.args.h1)
		self.wid_wEmbed.weight.requires_grad = True
		self.ANRS_RatingPred = ANRS_RatingPred(logger, args)




	def forward(self, batch_uid, batch_iid, verbose = 0):

		# Input
		batch_userDoc = self.uid_userDoc(batch_uid)
		batch_itemDoc = self.iid_itemDoc(batch_iid)
		rating_pred = self.ANRS_RatingPred(batch_userDoc, batch_itemDoc, verbose = verbose)


		if(verbose > 0):
			tqdm.write("\n[Final Output of {}] rating_pred: {}\n".format( self.args.model, rating_pred.size() ))

		return rating_pred


