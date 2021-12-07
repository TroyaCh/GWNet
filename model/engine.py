import torch.optim as optim
from model.gwn_model import *
import utils.util as util
class trainer():
    def __init__(self, scaler, supports, aptinit, gcn_bool=True, addaptadj=False, in_dim=1, seq_length=30, num_nodes=90, nhid=32, 
                 kernel_size=4, blocks=12, layers=2, walk_order=2, dropout=0, lrate=0.001, lrate_step=10, wdecay=0.0001, device='cuda:0'):
        
        # Initialize model.
        self.model = gwnet(device=device, num_nodes=num_nodes, dropout=dropout, supports=supports, gcn_bool=gcn_bool, 
                           addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, 
                           dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16, kernel_size=kernel_size, 
                           blocks=blocks, layers=layers, walk_order=walk_order)
        
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5
        # Add learning rate decay.
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=lrate_step, gamma=0.1)

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
