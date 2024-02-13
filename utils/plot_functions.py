import numpy as np
import matplotlib.pyplot as plt
import math
import os


def plot_predictions(epochs,log_dir, dataset_dir, train_mae ,train_smape , train_rmse, valid_mae, valid_smape, valid_rmse, output_name='outputs', CH=[1, 15], NSample=0, save_figure=True):
    '''
    Visualize model predictions. 
    
    log_dir: directory where model outputs were saved to
    dataset_dir: directory where data samples were saved to
    NROI: plot predictions from NROI[0] to NROI[1]
    NSample: number of test sample    
    '''

    test_data = np.load(dataset_dir + '/test.npz')
    outputs = np.load(os.path.join(log_dir, output_name) + '.npz')
    outputs_predicitons = outputs['predictions'].transpose((1,0,2))  # To have shape (samples, time, CH)
    outputs_groundtruth = outputs['groundtruth'].transpose((1,0,2))
    
    input_len=test_data['x'].shape[1]
    out= outputs_predicitons.shape[1]              #outputs_predicitons.shape[1]
    
    if out < 64:
        horizon= 32
    elif out == 64:
        horizon= 64
    else: horizon= 96

    for file_num in range(3):
        start_channel = file_num * 15 + 1
        end_channel = (file_num + 1) * 15

        figurename=('GWN_predictions_test_sample_{}'.format(NSample)) 
        figurename= figurename + '_channel{}-{}'.format(start_channel,end_channel)
        Nsubfigs = 15 #CH[1]-CH[0]+1
        fig=plt.figure(num=figurename, figsize=(27, Nsubfigs)) 
        fig.subplots_adjust(top=0.93, right = 0.88, left = 0.12, wspace = 0.15,
                            hspace=0.4, bottom = 0.07 )
        
        t_in = np.linspace(input_len - 30, input_len, 30)
        t_out= np.linspace(input_len+1,input_len+horizon,horizon)
        
        Nrows= math.ceil((Nsubfigs)/3)
            
        # Creat plots.
        for nCH in range(start_channel - 1, end_channel):

            ax = fig.add_subplot(Nrows, 3, nCH - start_channel + 1 + 1)

            plt.plot(t_in, test_data['x'][NSample,-30:,nCH,0], linestyle='-', marker='o', linewidth=3,
                    markersize='2.3', markeredgecolor='black', color=(0.129, 0.639, 0.588, 0.549))
            plt.plot(t_out, outputs_groundtruth[NSample,0:horizon,nCH], linestyle='-', marker='o', linewidth=3,
                    markersize='2.3', markeredgecolor='black', color=(0.129, 0.639, 0.588, 0.549), label='Truth')
            plt.plot(t_out, outputs_predicitons[NSample,0:horizon,nCH], linestyle='-', marker='o',
                    markersize='2.3', markeredgecolor='black', color='indigo', label='Prediction')
            plt.axvline(x=(input_len+0.5), color='black', linewidth='0.5')

            plt.legend()
            plt.title('Channel #{}'.format(nCH+1))

            ax.set_xlabel('Time')
            ax.set_ylabel('Signal')
        
        plt.suptitle('Predictions (Sample #{})'.format(NSample)) 

        if save_figure:
            print('Save figure in: ', log_dir)  
            plt.savefig(os.path.join(log_dir, figurename + '_' + output_name + '{}.png'.format(file_num)), dpi=100)
            
        plt.close()    

    # Create a list of metric names and corresponding data
    metrics = {
        'MAE': {'train': train_mae, 'valid': valid_mae},
        'sMAPE': {'train': train_smape, 'valid': valid_smape},
        'RMSE': {'train': train_rmse, 'valid':valid_rmse}
    }
    epochs = np.arange(1, epochs+1)

    # Loop through each metric
    for metric_name, metric_data in metrics.items():
        # Create a new figure for each metric
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot train and validation data
        ax.plot(epochs, metric_data['train'], label='Train ', color='blue')
        ax.plot(epochs, metric_data['valid'], label='Validation ', color='orange')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('loss')
        ax.legend()
        ax.set_title('Train vs Validation - {}'.format(metric_name))
        
        if save_figure:
            print('Save figure in: ', log_dir)  
            # Save the plot as a PNG file
            plt.savefig(os.path.join(log_dir, '{}_plot.png'.format(metric_name.lower())),bbox_inches='tight' ,dpi=100)
            #plt.savefig('{}_plot.png'.format(metric_name.lower()), bbox_inches='tight')
            #plt.close()  # Close the current figure to release resources

            print("Plots saved as PNG files.")

