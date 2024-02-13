import torch
import numpy as np
import time
import os
import utils.util as util
import matplotlib.pyplot as plt
from utils.plot_functions import plot_predictions
from model.engine import trainer
import yaml
import argparse
import torch.distributed as dist


parser = argparse.ArgumentParser()
parser.add_argument('--config',type=str,help='config path')
args_ = parser.parse_args()

config_file_path = args_.config

with open (config_file_path,'r') as config_file:
    config = yaml.safe_load(config_file)

# Create an argparse Namespace with the loaded parameters
args = argparse.Namespace(**config)
# def setup(rank, world_size):
#     #initialize the process group
#     dist.init_process_group("nccl", rank= rank, world_size=world_size)

def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    
    device = torch.device(args.device)
    adj_mx_train = util.load_adj(args.train_support, args.adjtype) # ssymn adj or asymatric adj
    adj_mx_val = util.load_adj(args.val_support, args.adjtype)
    adj_mx_test = util.load_adj(args.test_support, args.adjtype)

    nbh_adj_mx = util.load_adj(args.adjdata_nbh, args.adjtype) # neighbour info

    if args.save_dir == None:  # Create directory for outputs if  not specified.
        args.save_dir = './results/GWN_gcn{}_nhid{}_lr{}_{}/'.format(args.gcn_bool, args.nhid,
                                                                    args.learning_rate, time.strftime('%m%d%H%M%S'))
    if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']# mean, std initialized object
    min_max_scaler = dataloader['min_max_scaler']
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")

    # supports_train = [torch.tensor(i).to(device) for i in adj_mx_train]
    # supports_val = [torch.tensor(i).to(device) for i in adj_mx_val]
    # supports_test = [torch.tensor(i).to(device) for i in adj_mx_test]

    # supports_nbh =[torch.tensor(i, dtype=torch.float32).to(device) for i in nbh_adj_mx]

    supports_train = [torch.tensor(i, dtype=torch.float32).to(device) for i in adj_mx_train]
    supports_val = [torch.tensor(i, dtype=torch.float32).to(device) for i in adj_mx_val]
    supports_test = [torch.tensor(i, dtype=torch.float32).to(device) for i in adj_mx_test]

    supports_nbh =[torch.tensor(i).to(device) for i in nbh_adj_mx]
  
    logger = util.get_logger(args.save_dir, __name__, 'info.log', level='INFO')

    logger.info(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None


    train_engine = trainer(scaler=scaler,min_max_scaler=min_max_scaler, supports=supports_train, supports_nbh = supports_nbh, aptinit=adjinit, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, in_dim=args.in_dim, 
                     seq_length=args.seq_length, num_nodes=args.num_nodes, nhid=args.nhid, kernel_size=args.kernel_size, blocks=args.blocks, 
                     layers=args.layers, walk_order=args.walk_order, dropout=args.dropout, lrate=args.learning_rate, 
                     lrate_step=args.learning_rate_step, wdecay=args.weight_decay, device=device)
    
    val_engine = trainer(scaler=scaler, min_max_scaler=min_max_scaler,supports=supports_val, supports_nbh = supports_nbh, aptinit=adjinit, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, in_dim=args.in_dim, 
                     seq_length=args.seq_length, num_nodes=args.num_nodes, nhid=args.nhid, kernel_size=args.kernel_size, blocks=args.blocks, 
                     layers=args.layers, walk_order=args.walk_order,dropout=args.dropout, lrate=args.learning_rate, 
                     lrate_step=args.learning_rate_step, wdecay=args.weight_decay, device=device)
    
    test_engine = trainer(scaler=scaler, min_max_scaler=min_max_scaler,supports=supports_test, supports_nbh = supports_nbh, aptinit=adjinit, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, in_dim=args.in_dim, 
                     seq_length=args.seq_length, num_nodes=args.num_nodes, nhid=args.nhid, kernel_size=args.kernel_size, blocks=args.blocks, 
                     layers=args.layers, walk_order=args.walk_order, dropout=args.dropout, lrate=args.learning_rate, 
                     lrate_step=args.learning_rate_step, wdecay=args.weight_decay, device=device)
    
    # Print the model summary
    train_engine.print_model_summary()

    # Training.
    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    ep_tr_loss=[]
    ep_tr_smape=[]
    ep_tr_rmse=[]
    ep_vl_loss=[]
    ep_vl_smape=[]
    ep_vl_rmse=[]
    for i in range(1,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_smape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            #print("trainx",trainx.shape) #[64, 30, 90, 1]
            trainx= trainx.transpose(1, 3) # dim swap
            trainy = torch.Tensor(y).to(device)
            #print(trainy.shape)
            trainy = trainy.transpose(1, 3)
            metrics = train_engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_smape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train SMAPE: {:.4f}, Train RMSE: {:.4f}, Lr: {:.6f}'
                logger.info(log.format(iter, train_loss[-1], train_smape[-1], train_rmse[-1], train_engine.optimizer.param_groups[0]['lr']))
                torch.save(train_engine.model.state_dict(), args.save_dir+"_epoch_"+str(i)+".pth")
        t2 = time.time()
        train_time.append(t2-t1)
        train_engine.scheduler.step()

        # Validation.
        valid_loss = []
        valid_smape = []
        valid_rmse = []


        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            val_engine.model.load_state_dict(torch.load(args.save_dir+"_epoch_"+str(i)+".pth")) 
            metrics = val_engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_smape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        #logger.info(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_smape = np.mean(train_smape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_smape = np.mean(valid_smape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        ep_tr_loss.append(mtrain_loss)
        ep_tr_smape.append(mtrain_smape)
        ep_tr_rmse.append(mtrain_rmse)
        ep_vl_loss.append(mvalid_loss)
        ep_vl_smape.append(mvalid_smape)
        ep_vl_rmse.append(mvalid_rmse)


        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train SMAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid SMAPE: {:.4f}, Valid RMSE: {:.4f}, Lr: {:.6f}, Training Time: {:.4f} min/epoch'
        logger.info(log.format(i, mtrain_loss, mtrain_smape, mtrain_rmse, mvalid_loss, mvalid_smape, mvalid_rmse, train_engine.optimizer.param_groups[0]['lr'], (t2 - t1)/60))
        torch.save(train_engine.model.state_dict(), args.save_dir+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,4))+".pth")
    logger.info("Average Training Time: {:.4f} min/epoch".format(np.mean(train_time)/60))
    logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    

    # Testing.
    bestid = np.argmin(his_loss)
    test_engine.model.load_state_dict(torch.load(args.save_dir+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],4))+".pth"))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = test_engine.model(testx).transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))


    amae = []
    asmape = []
    armse = []
    for i in range(args.seq_length):
        pred = scaler.inverse_transform_3d(min_max_scaler.inverse_transform_3d(yhat[:,:,i]))
        real = realy[:,:,i]
        print("real",real.shape)
        print("pred", pred.shape)
    
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test SMAPE: {:.4f}, Test RMSE: {:.4f}'
        logger.info(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        asmape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over {} horizons, Test MAE: {:.4f}, Test SMAPE: {:.4f}, Test RMSE: {:.4f}'
    logger.info(log.format(args.seq_length, np.mean(amae),np.mean(asmape),np.mean(armse)))
    torch.save(test_engine.model.state_dict(), args.save_dir+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],4))+".pth")
    
    yhat= scaler.inverse_transform(min_max_scaler.inverse_transform(yhat.transpose(1,2).unsqueeze(-1)))
    # Save model outputs.
    yhat_np = np.array(yhat.cpu().squeeze(-1).transpose(0,1))  # Convert to numpy arrray and reshape to (horizon, samples, NROIs)
    realy_np = np.array(realy.cpu()).transpose(2,0,1)
    
    if args.save_predictions:
        print('Save outputs in: ', args.save_dir)
        np.savez(args.save_dir + args.output_name, 
                 predictions=yhat_np,  # Recover original scaling.
                 groundtruth=realy_np)  
        # Plot predictions and save plots.
        plot_predictions(args.epochs, args.save_dir, args.data, ep_tr_loss, ep_tr_smape, ep_tr_rmse, ep_vl_loss, ep_vl_smape, ep_vl_rmse, output_name=args.output_name )


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}s".format(t2-t1))
