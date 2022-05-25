import json
import torch, random

from client import *
from Process.process import *
from Process.rand5fold import *
from Process.rand5fold import *
from Process.evaluate import *
	
from model import *

from torch_geometric.data import DataLoader
import torch.nn.functional as F
import numpy as np

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

class Server(object):
	
	def __init__(self, conf, eval_dataset):
	
		self.conf = conf 
		
		self.global_model = Net(5000,64,64,5).to(device)
		
		self.eval_loader = DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True, num_workers=5)
		
	def model_aggregate(self, weight_accumulator):
		for name, data in self.global_model.state_dict().items():
			
			update_per_layer = weight_accumulator[name] * 1/(self.conf["k"])
			
			if data.type() != update_per_layer.type():
				data.add_(update_per_layer.to(th.int64))
			else:
				data.add_(update_per_layer)
				
	def model_eval(self):
		self.global_model.eval()
        
		temp_val_losses = []
		temp_val_accs = []
		temp_val_F1, temp_val_F2, temp_val_F3, temp_val_F4 = [], [], [], []
            
		test_loader = self.eval_loader
		for Batch_data in test_loader:
		    Batch_data.to(device)
		    val_out = self.global_model(Batch_data)
		    val_loss = F.nll_loss(val_out, Batch_data.y)
		    temp_val_losses.append(val_loss.item())
		    _, val_pred = val_out.max(dim=-1)
		    Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, Batch_data.y)
		    temp_val_accs.append(Acc_all)
		    temp_val_F1.append(F1)
		    temp_val_F2.append(F2)
		    temp_val_F3.append(F3)
		    temp_val_F4.append(F4)

		los = np.mean(temp_val_losses)
		accs = np.mean(temp_val_accs)
		F1 = np.mean(temp_val_F1)
		F2 = np.mean(temp_val_F2)
		F3 = np.mean(temp_val_F3)
		F4 = np.mean(temp_val_F4)
		
		return los,accs,F1,F2,F3,F4

if __name__ == '__main__':

	with open('./conf.json', 'r') as f:
		conf = json.load(f)	

	dataname="Twitter16"
	treeDic=loadTree(dataname)
	x_test,x_train = load5foldData(dataname)
	droprate=0.2
    
	train_datasets, eval_datasets = loadBiData(dataname, treeDic, x_train, x_test, droprate)
    
	ser = Server(conf, eval_datasets)
	clients = []
	
	for c in range(conf["no_models"]):
		clients.append(Client(conf, train_datasets, c))
	Los = []
	Accs = []
	for e in range(conf["global_epochs"]):		
		print('\n global_epoch: %d' % (e))
		candidates = random.sample(clients, conf["k"])		
		weight_accumulator = {}		
		for name, params in ser.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)		
		for c in candidates:
			diff = c.local_train(ser.global_model)			
			for name, params in ser.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name])				
		ser.model_aggregate(weight_accumulator)
		ser.model_eval()
		los,accs,F1,F2,F3,F4 = ser.model_eval()
		Los.append(los)
		Accs.append(accs)
		print("\n Epoch %d, acc: %f, loss: %f\n" % (e, accs, los))
	print('Los:',Los)
	print('Accs:',Accs)

