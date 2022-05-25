from model import *
import torch as th
from torch_geometric.data import DataLoader
import torch.nn.functional as F

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

class Client(object):

	def __init__(self, conf, train_dataset, id = -1):
		
		self.conf = conf
		
		self.local_model = Net(5000,64,64,5).to(device)
		
		self.client_id = id
		
		self.train_dataset = train_dataset
		
		all_range = list(range(len(self.train_dataset)))
		data_len = int(len(self.train_dataset) / self.conf['no_models'])
		train_indices = all_range[id * data_len: (id + 1) * data_len]

		self.train_loader = DataLoader(self.train_dataset, batch_size=conf["batch_size"], 
									sampler=th.utils.data.sampler.SubsetRandomSampler(train_indices), num_workers=5)
									
		
	def local_train(self, model):

		for name, param in model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())

		optimizer = th.optim.Adam(self.local_model.parameters(),lr=self.conf['lr'],weight_decay=self.conf['weight_decay'])

		self.local_model.train()
		for e in range(self.conf["local_epochs"]):
			train_loader = self.train_loader
            
			for Batch_data in train_loader:
			    Batch_data.to(device)
			    out_labels= self.local_model(Batch_data)
			    loss=F.nll_loss(out_labels,Batch_data.y)
			    optimizer.zero_grad()
			    loss.backward()
			    optimizer.step()

		diff = dict()

		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - model.state_dict()[name])
			
		return diff
		
