import numpy as np
from mne_connectivity import spectral_connectivity_epochs


def SpectralConnectivity(x,sfreq ):
    fmin=(8) 
    fmax=(12)
    
    #todo-
    # shape of x = no_win, horizon, nodes, feature  _ , 30,64,1
    # squeeze shape to _,30,64
    # swape _ ,30,64 --> _,64,30
    # 
    # spectral_connectivity_epochs takes   shape=(n_epochs, n_signals, n_times) 
    
    x= np.squeeze(x)
    x = np.transpose(x,(0,2,1))
    
    S = spectral_connectivity_epochs(x, sfreq=sfreq, method='coh', mode='multitaper', fmin=fmin, fmax=fmax, faverage=True)

    matrix= S.get_data(output= 'dense')
    matrix =np.squeeze(matrix)

    # Create a copy of the matrix
    S = matrix.copy()

    # Fill the diagonal of the copy with 1
    np.fill_diagonal(S, 1)

    # Calculate S + S.T - np.diag(S.diagonal())
    S = S + S.T - np.diag(S.diagonal())

    return S