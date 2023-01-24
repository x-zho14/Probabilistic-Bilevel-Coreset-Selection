# Probabilistic-Bilevel-Coreset-Selection

## Requirements:

```
Pytorch 1.7
Python 3.7.7
CUDA Version 10.1
pyyaml 5.3.1
tensorboard 2.2.1
torchvision 0.5.0
tqdm 4.50.2
```

## Command
Below are the commands for replicating the results of coreset selection and pixel selection experiments.

```bash
CUDA_VISIBLE_DEVICES=0  python cnn_mnist_probability_1step_pixel_shared_rein.py --inner_lr 5e-3 --inner_optim adam --test_freq 10 --outer_lr 1e-1 --max_outer_iter 2000 --limit 1000 --coreset_size 100 --div_tol 2 --epoch_converge 300 --runs_name outer1e-1_limit1000_size100_shared --wandb --vr --K 5 --clip_grad --test_freq 5 --start_coreset_size 784 --iterative

CUDA_VISIBLE_DEVICES=0  python cnn_mnist_probability_1step_reinforce.py --limit 1000  --inner_lr 5e-3 --inner_optim Adam --outer_lr 5e-2 --coreset_size 200 --runs_name size200 --wandb --iterative --project coreset_size_iterative  --epoch_converge 300 --test_freq 10 --max_outer_iter 2000 --wandb --vr --K 5 --clip_grad --test_freq 5
```
## Cite
If you find this implementation is helpful to your work, please cite 

```BibTeX
@inproceedings{coreset,
  title={Probabilistic Bilevel Coreset Selection},
  author={Zhou, Xiao and Pi, Renjie and Zhang, Weizhong and Lin, Yong and Zhang, Tong},
  booktitle={International Conference on Machine Learning},
  year={2022},
  organization={PMLR}
}

```


