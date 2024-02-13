import numpy as np 
import scipy.io as sio
import os 
import argparse
from utils.spectral_conn import SpectralConnectivity

def prepare_train_val_test(input_dir, output_dir, adj_dir, x_size =256, y_size=256, 
                           overlap = False, sfreq=512, train_prop=0.8, test_prop=0.05, save_data=True):
    
    input_files = os.listdir(input_dir)
    print(input_files)
    Overlap = overlap
    x, y = [],[]
    all =[]
    #t=0
    for input_file in input_files:
        filename = os.path.join(input_dir, input_file)
        subject = sio.loadmat(filename)
        # t+=1
        # print(t)
        subject = np.asarray(subject['dt_512_1s'], dtype=float) #dt256, dt_512_1s
        #print(filename)
        #subject = np.asarray(subject['dt_{}_1s'.format(x_size)], dtype=float)  # shape (65 in 1 index, 224000 in 0 index)
        num_channels, num_samples = subject.shape
        print(subject.shape)

        subject = subject.T  # (224000, 65)
        print(Overlap)
        if Overlap is True:
            print('0.5 overlap')
            overlap = x_size // 2  # 75% overlap
            i = 0
            while i + x_size + y_size <= num_samples:
                window = subject[i:i+x_size, :]  # yml
                horizon = subject[i+x_size:i+x_size+y_size, :]
                x.append(window)
                y.append(horizon)
                
                a= subject[i:i+x_size+y_size, :]

                all.append(a)

                i += overlap
                
        else:    
            print('w/o overlap')    
            Range = (num_samples-x_size)//x_size
                # Iterate over the samples using the sliding window approach
            for i in range(Range):
                window = subject[i:i+x_size,:]  # yml
                horizon = subject[i+x_size:i+x_size+y_size,:]
                x.append(window)
                y.append(horizon)

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)


    # Add feature dimension.
    x = np.expand_dims(x, axis=-1)
    y = np.expand_dims(y, axis=-1)

    all = np.stack(all, axis=0)

    # Compute number of train, val, test samples.
    num_snippets = x.shape[0]
    num_test = round(num_snippets * test_prop)
    num_train = round(num_snippets * train_prop)
    num_val = num_snippets - num_test - num_train    
    
    # Split data.
    x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], [] 
    # Train.
    x_train.append(x[:num_train]), y_train.append(y[:num_train])
    # Val.
    x_val.append(x[num_train: num_train + num_val]), y_val.append(y[num_train: num_train + num_val])       
    # Test.
    x_test.append(x[-num_test:]), y_test.append(y[-num_test:])
    
    x_train = np.concatenate(x_train, axis=0)  # Concatenate all samples along first dimension.    
    y_train = np.concatenate(y_train, axis=0)
    x_val = np.concatenate(x_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    matrix_train = SpectralConnectivity(x_train, sfreq)
    matrix_val = SpectralConnectivity(x_val, sfreq)
    matrix_test = SpectralConnectivity(x_test, sfreq)

    print('### SAMPLES ###')
    print('In total:    {:5} training samples, {:5} validation samples, {:5} testing samples.'.format(x_train.shape[0], x_val.shape[0], x_test.shape[0]))
    

     # Save results.
    if save_data:
        print('### SAVE DATA ###')
        print('Save in: ' + output_dir)
        for cat in ["train", "val", "test"]:
            _w, _h = locals()["x_" + cat], locals()["y_" + cat]
            print(cat, "window: ", _w.shape, "horizon:", _h.shape)
            np.savez_compressed(
            os.path.join(output_dir, "%s.npz" % cat),
            x=_w,
            y=_h,
            )
        print("matrix save in:" + adj_dir)
        for cat in ["train", "val", "test"]:
            _m = locals()["matrix_" + cat]
            np.save(os.path.join(adj_dir,"spectral_matrix_%s.npy" % cat), _m),
   
        np.save(os.path.join(output_dir,"all.npy" 
                             ),all),
                
    return x_train, y_train, x_val, y_val, x_test, y_test

def main(args):
    print("preparing data...")
    
    prepare_train_val_test(input_dir=args.input_dir, output_dir= args.output_dir, adj_dir= args.adj_dir, x_size=args.window, y_size=args.horizon , overlap =args.overlap )
    print(args.overlap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type = str, default="./data/")
    parser.add_argument("--output_dir", type = str, default="./data/")
    parser.add_argument("--adj_dir", type=str, default=".data/adj_mx_256_2s_015s_05")
    parser.add_argument("--window", type=int, help='window size to train')
    parser.add_argument("--horizon",type=int, help='horizon size to be predict')
    parser.add_argument("--overlap", type=bool, default=False, help='if overlap is True 0.5 overlap')

    args= parser.parse_args()
    main(args)





