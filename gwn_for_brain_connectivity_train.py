import torch
import numpy as np
import argparse
import time
import os
import utils.util as util
import matplotlib.pyplot as plt
from utils.plot_functions import plot_predictions
from model.engine import trainer


parser = argparse.ArgumentParser()
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
parser.add_argument('--save_dir',type=str,default=None,help='directory for saving logs and weights')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--save_predictions',type=bool,default=True,help='save model outputs')
parser.add_argument('--output_name',type=str,default='outputs',help='name of model outputs')


args = parser.parse_args()


def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)
    adj_mx = util.load_adj(args.adjdata, args.adjtype)
    if args.save_dir == None:  # Create directory for outputs if  not specified.
        args.save_dir = './results/GWN_gcn{}_{}_nhid{}_lr{}_{}/'.format(args.gcn_bool, args.walk_order, args.nhid,
                                                                    args.learning_rate, time.strftime('%m%d%H%M%S'))
    if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    logger = util.get_logger(args.save_dir, __name__, 'info.log', level='INFO')

    logger.info(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None


    engine = trainer(scaler=scaler, supports=supports, aptinit=adjinit, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, in_dim=args.in_dim, 
                     seq_length=args.seq_length, num_nodes=args.num_nodes, nhid=args.nhid, kernel_size=args.kernel_size, blocks=args.blocks, 
                     layers=args.layers, walk_order=args.walk_order, dropout=args.dropout, lrate=args.learning_rate, 
                     lrate_step=args.learning_rate_step, wdecay=args.weight_decay, device=device)
    

    # Training.
    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    for i in range(1,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Lr: {:.6f}'
                logger.info(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1], engine.optimizer.param_groups[0]['lr']))
        t2 = time.time()
        train_time.append(t2-t1)
        engine.scheduler.step()

        # Validation.
        valid_loss = []
        valid_mape = []
        valid_rmse = []


        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        #logger.info(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Lr: {:.6f}, Training Time: {:.4f} min/epoch'
        logger.info(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, engine.optimizer.param_groups[0]['lr'], (t2 - t1)/60))
        torch.save(engine.model.state_dict(), args.save_dir+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,4))+".pth")
    logger.info("Average Training Time: {:.4f} min/epoch".format(np.mean(train_time)/60))
    logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    

    # Testing.
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save_dir+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],4))+".pth"))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))


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
    torch.save(engine.model.state_dict(), args.save_dir+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],4))+".pth")
    
    # Save model outputs.
    yhat_np = np.array(yhat.cpu()).transpose(2,0,1)  # Convert to numpy arrray and reshape to (horizon, samples, NROIs)
    realy_np = np.array(realy.cpu()).transpose(2,0,1)
    
    if args.save_predictions:
        print('Save outputs in: ', args.save_dir)
        np.savez(args.save_dir + args.output_name, 
                 predictions=scaler.inverse_transform(yhat_np),  # Recover original scaling.
                 groundtruth=realy_np)  
        # Plot predictions and save plots.
        plot_predictions(args.save_dir, args.data, output_name=args.output_name)


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
