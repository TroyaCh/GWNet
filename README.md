# Graph WaveNet for brain connectivity

<img src="https://github.com/simonvino/GraphWaveNet_brain_connectivity/blob/main/figures/GWN_for_brain_connectivity.png" width="800">


This is the implementation of the Graph WaveNet model used in our manuscript:

S. Wein , A. Schüller, A. M. Tome, W. M. Malloni, M. W. Greenlee, and E. W.
Lang,
[Modeling Spatio-Temporal Dynamics in Brain Networks: A Comparison of Graph Neural Network Architectures](https://arxiv.org/abs/2112.04266).

The implementation is based on the [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet) proposed by:

Z. Wu, S. Pan, G. Long, J. Jiang, C. Zhang, [Graph WaveNet for Deep Spatial-Temporal Graph Modeling](https://arxiv.org/abs/1906.00121), IJCAI 2019.


## Requirements

- pytroch>=1.00
- scipy>=0.19.0
- numpy>=1.12.1

Also a conda *environment.yml* file is provided. The environment can be installed with:

```
conda env create -f environment.yml
```

## Run demo version

A short demo version is included in this repository, which can serve as a template to process your own MRI data. Artificial fMRI data is provided in the directory ``` MRI_data/fMRI_sessions/ ``` and the artificial timecourses have the shape ``` (nodes,time) ```. 
The adjacency matrix in form of the structural connectivity (SC) between brain regions can be stored in ``` MRI_data/SC_matrix/ ```. An artificial SC matrix with shape ``` (nodes,nodes) ``` is also provided in this demo version.

The training samples can be generated from the subject session data by running: 

```
python generate_samples.py --input_dir=./MRI_data/fMRI_sessions/ --output_dir=./MRI_data/training_samples
```

The model can then be trained by running:

```
python gwn_for_brain_connectivity_train.py --data ./MRI_data/training_samples --save_predictions True
```


A Jupyter Notebook version is provided, which can be directly run in Google Colab with:

> https://colab.research.google.com/github/simonvino/GraphWaveNet_brain_connectivity/blob/main/gwn_for_brain_connectivity_colab_demo.ipynb



## Data availability

Preprocessed fMRI and DTI data from Human Connectome Project data is publicly available under: https://db.humanconnectome.org.

A nice tutorial on white matter tracktography for creating a SC matrix is available under: https://osf.io/fkyht/. 

## Citations

Our arXiv manuscript can be cited as: 

```
@misc{Wein2021GNNs_bc,
      title={Modeling Spatio-Temporal Dynamics in Brain Networks: A Comparison of Graph Neural Network Architectures}, 
      author={Simon Wein and Alina Schüller and Ana Maria Tomé and Wilhelm M. Malloni and Mark W. Greenlee and Elmar W. Lang},
      year={2021},
      eprint={2112.04266},
      archivePrefix={arXiv},
      primaryClass={q-bio.NC}
}
```

And the model architecture was originally proposed by Wu et al.:

```
@inproceedings{Wu2019_GWN_traffic,
  title={Graph WaveNet for Deep Spatial-Temporal Graph Modeling},
  author={Wu, Zonghan and Pan, Shirui and Long, Guodong and Jiang, Jing and Zhang, Chengqi},
  booktitle={Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI-19)},
  year={2019}
}
```


