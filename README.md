# Towards Evaluating and Training Verifiably Robust Neural Networks
This repo intends to release code for our work: 


Zhaoyang Lyu, Minghao Guo, Tong Wu, Guodong Xu, Kehuan Zhang, Dahua Lin, ["Towards Evaluating and Training Verifiably Robust Neural Networks"](https://arxiv.org/abs/2104.00447), CVPR 2021 (Oral).


The appendix of our paper is in the file `_appendix.pdf`.

Updates
----------------------------------------------------------------
- Jun 26, 2021: Initial release. Release the codes for experiments on MNIST dataset.


Setup
----------------------------------------------------------------
The code is tested with python 3.8.5, Pytorch 1.6.0 and CUDA 10.1. Run the following conda command to create a environment with all the requirements:
```
conda create --name verify --file conda_spec.txt
conda activate verify 
```

Train Verifiably Robust Neural Networks
----------------------------------------------------------------
To train verifiably robust neural networks, just call the file `train.py` with a config file. 
For example, if you want to use CROWN-IBP to train a DM-large network with ParamRamp activation on MNIST dataset, run the following command:
``` 
python train.py --config exp_configs/mnist/crown-ibp/mnist_dm-large_crown-ibp_ParamRamp(0.01~0)_kappa_1-0.5.json
```
`ParamRamp(0.01~0)` in the file name means that we adopt a deceasing schedule of the neg-slope of the ParamRamp activation, i.e., the neg-slope starts with 0.01 and gradually decrease to 0 in the training process. Similarly, `kappa_1-0.5` means that we adopt a deceasing schedule of the hyper parameter **kappa** described in the paper.

All experiments reported in our paper has a corresponding config file in the folder `exp_configs`. You should easily find the corresponding config file for a specific experiment based on their names.

For each experiment, the results will be saved to the path specified by **models_path** in the corresponding config file.
For MNIST dataset, the model is trained at **epsilon=0.4** and tested at **epsilon=0.2, 0.3, 0.4** respectively. We save the model with the lowest IBP verified errors in the training process.

The experiments are run on GPUs by default. You can change the **device** in **training_params** in the config file to set the index of the GPU where you want to run the experiments. You could also set **multi_gpu** to true and specify the **device_ids** if you want to use multiple GPUs for training.

Evaluate Trained Networks
----------------------------------------------------------------
We empirically evaluted the trained networks by 200-step PGD attacks with 10 random starts. We also compute IBP, CROWN-IBP, LBP, and CROWN-LBP verified errors for the trained networks.
To evaluate the above trained network at **epsilon=0.2**, run the following command:
```
python eval_models.py --epsilon 0.2 --config exp_configs/mnist/crown-ibp/mnist_dm-large_crown-ibp_ParamRamp(0.01~0)_kappa_1-0.5.json 
```
This command will automatically load the trained network and evaluate it using PGD attacks and compute IBP, CROWN-IBP, LBP, and CROWN-LBP verified errors for the network. You can also change some arguments for evaluation specified in the argparser in the file `eval_models.py`.


Acknowledgements
----------------------------------------------------------------
This repo is adapted from the Repo https://github.com/huanzhang12/CROWN-IBP. 
