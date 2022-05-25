from model import *
import torch as th
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from Process.evaluate import *
import numpy as np
	

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

class Client(object):

	def __init__(self, conf, train_dataset,eval_dataset):
		
		self.conf = conf
		
		self.local_model = Net(5000,64,64,5).to(device)
		
		self.train_dataset = train_dataset
		self.eval_dataset = eval_dataset
		
		self.train_loader = DataLoader(self.train_dataset, batch_size=conf["batch_size"], shuffle=True, num_workers=5)
		self.eval_loader = DataLoader(self.eval_dataset, batch_size=conf["batch_size"], shuffle=True, num_workers=5)
	
	def local_train(self, model,localstat):

		for name, param in model.state_dict().items():
			lamda = 1
			self.local_model.state_dict()[name].copy_(localstat[name].clone()*(1-lamda))
			update_per_layer = param*lamda
			
			if self.local_model.state_dict()[name].type() != update_per_layer.type():
				self.local_model.state_dict()[name].add_(update_per_layer.to(th.int64))
			else:
				self.local_model.state_dict()[name].add_(update_per_layer)

		optimizer =th.optim.Adam(self.local_model.parameters(),lr=self.conf['lr'],weight_decay=self.conf['weight_decay'])

		self.local_model.train()
		temp_val_losses = []
		temp_val_accs = []
		for e in range(self.conf["local_epochs"]):
			train_loader = self.train_loader
            
			for Batch_data in train_loader:
			    Batch_data.to(device)
			    out_labels= self.local_model(Batch_data)
			    loss=F.nll_loss(out_labels,Batch_data.y)
			    optimizer.zero_grad()
			    loss.backward()
			    optimizer.step()
			    
			self.local_model.eval()

			temp_val_F1, temp_val_F2, temp_val_F3, temp_val_F4 = [], [], [], []
			test_loader = self.eval_loader
			for Batch_data in test_loader:
		 	   Batch_data.to(device)
		 	   val_out = self.local_model(Batch_data)
		 	   val_loss = F.nll_loss(val_out, Batch_data.y)
		 	   temp_val_losses.append(val_loss.item())
		 	   _, val_pred = val_out.max(dim=-1)
		 	   Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(val_pred, Batch_data.y)
		 	   temp_val_accs.append(Acc_all)
		 	   temp_val_F1.append(F1)
		 	   temp_val_F2.append(F2)
		 	   temp_val_F3.append(F3)
		 	   temp_val_F4.append(F4)


		F1 = np.mean(temp_val_F1)
		F2 = np.mean(temp_val_F2)
		F3 = np.mean(temp_val_F3)
		F4 = np.mean(temp_val_F4)

		los = np.mean(temp_val_losses)
		accs = np.mean(temp_val_accs)
		print("\n acc: %f, loss: %f\n" % (accs, los))
		 	
		diff = dict()

		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - model.state_dict()[name])
		localstat= self.local_model.state_dict()	
		return diff,los,accs,localstat,F1,F2,F3,F4
		
