import utils.util as util
import argparse
from model.gwn_model import *
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from utils.plot_functions import plot_predictions


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir',type=str,default=None,help='directory where the model is stored in')
parser.add_argument('--checkpoint_model',type=str,default=None,help='name of model to restore (.pth)')
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='./MRI_data/training_samples',help='data path')
parser.add_argument('--adjdata',type=str,default='./MRI_data/SC_matrix/artificial_SC_matrix.npy',help='adj data path')
parser.add_argument('--adjtype',type=str,default='transition',help='adj type')
parser.add_argument('--gcn_bool',type=bool,default=True,help='whether to add graph convolution layer')
parser.add_argument('--aptonly',type=bool,default=False,help='whether only adaptive adj')
parser.add_argument('--addaptadj',type=bool,default=False,help='whether add adaptive adj')
parser.add_argument('--randomadj',type=bool,default=True,help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=30,help='length of predicted sequence')
parser.add_argument('--nhid',type=int,default=32,help='number of channels')
parser.add_argument('--kernel_size',type=int,default=4,help='kernel size')
parser.add_argument('--blocks',type=int,default=8,help='temporal convolution blocks')
parser.add_argument('--layers',type=int,default=2,help='temporal convolution layers')
parser.add_argument('--walk_order',type=int,default=2,help='walk order on graph')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=90,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--learning_rate_step',type=float,default=10,help='learning rate decay step')
parser.add_argument('--dropout',type=float,default=0,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=40,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--save_predictions',type=bool,default=True,help='save model outputs')
parser.add_argument('--output_name',type=str,default='outputs',help='name of model outputs')


args = parser.parse_args()


def main():
    if args.checkpoint_dir == None:  # Look for newest results if not specified.
        args.checkpoint_dir = max(glob.glob(os.path.join('./results', '*/')), key=os.path.getmtime)
    if args.checkpoint_model == None:  # Look for best model if not specified.
        checkpoint = max(glob.glob(os.path.join(args.checkpoint_dir, '*_best_*.pth')), key=os.path.getmtime)
    else:
        checkpoint = args.checkpoint_dir + args.checkpoint_model
    
    device = torch.device(args.device)
    adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    logger = util.get_logger(args.checkpoint_dir, __name__, 'info.log', level='INFO')
    
    logger.info(args)
    logger.info('Load: {}'.format(checkpoint))
    
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    model = gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit)
    model.to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    
    logger.info('Model has loaded successfully.')

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    min_max_scaler = dataloader['min_max_scaler']
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = model(testx).transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    amae = []
    amape = []
    armse = []
    for i in range(args.seq_length):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        logger.info(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    logger.info(log.format(args.seq_length, np.mean(amae),np.mean(amape),np.mean(armse)))
    
    # Save model outputs.
    yhat_np = np.array(yhat.cpu()).transpose(2,0,1)  # Convert to numpy arrray and reshape to (horizon, samples, NROIs)
    realy_np = np.array(realy.cpu()).transpose(2,0,1)
    
    if args.save_predictions:
        print('Save outputs in: ', args.checkpoint_dir)
        np.savez(os.path.join(args.checkpoint_dir, args.output_name), 
                 predictions=scaler.inverse_transform(yhat_np),  # Recover original scaling.
                 groundtruth=realy_np)  
        # Plot predictions and save plots.
        plot_predictions(args.checkpoint_dir, args.data, output_name=args.output_name)

if __name__ == "__main__":
    main()