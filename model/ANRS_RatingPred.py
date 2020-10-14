import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm



class ANRS_RatingPred(nn.Module):

	def __init__(self, logger, args):

		super(ANRS_RatingPred, self).__init__()

		self.logger = logger
		self.args = args


		# User/Item FC to learn the abstract user & item representations, respectively
		self.userFC = nn.Linear(self.args.num_aspects * self.args.h1, self.args.h1)
		self.itemFC = nn.Linear(self.args.num_aspects * self.args.h1, self.args.h1)

		# Dropout, using the specified dropout probability
		self.userFC_Dropout = nn.Dropout(p = self.args.dropout_rate)
		self.itemFC_Dropout = nn.Dropout(p = self.args.dropout_rate)

		# Dimensionality of the abstract user & item representations
		self.user_item_rep_dim = self.args.h1

		# Prediction Layer
		self.prediction = nn.Linear(2 * self.user_item_rep_dim, 1)

		# Initialize all weights using random uniform distribution from [-0.01, 0.01]
		self.userFC.weight.data.uniform_(0.00, 0.01)
		self.itemFC.weight.data.uniform_(0.00, 0.01)
		self.prediction.weight.data.uniform_(0.00, 0.01)



	# https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
	# def forward(self, userAspRep, itemAspRep, verbose = 1):
	def forward(self, userWordEmb, itemWordEmb, verbose = 1):
		n = userWordEmb.size(0)
		m = itemWordEmb.size(0)
		d = userWordEmb.size(1)

		x = userWordEmb
		y = itemWordEmb

		dist = torch.pow(torch.nn.functional.relu(x - y), 2).sum(1)

		if(verbose > 0):
			tqdm.write("\n[VOE_Loss Output] dist: {}".format( dist.size() ))
			tqdm.write("============================== =================================== ==============================\n")

		return dist


