# Deep Secret Sharing: Towards Preserving Model Confidentiality in DLaaS

## Installation

This repo was tested with Ubuntu 20.04.5 LTS, Python 3.10, and PyTorch 2.0.0. But it should be runnable with other PyTorch versions.

## Running

1. Download benchmark datasets into `/data`, if required. We test on CIFAR10, CIFAR100, CelebA, and TinyImageNet.

2. Run DSS on CIFAR10/CIFAR100 by `train32.py`. Run DSS on CelebA/TinyImageNet by `train64.py`. An example of running MinSim+ DSS scheme is given by:

    ```
    python train32.py --arch resnet20 --dataset cifar10 --n_party 2 --method 4 --lam 30 --alpha 0.1 
    ```

    where some important flags are explained as:
    - `--arch`: specify the model, []
    - `--dataset`: specify the dataset, []
    - `--n_party`: the number of parties (shares)
    - `--mod`: specify the or similarity measure, default: `cos`
      - `--mod cos`: Cosine similarity
      - `--mod jsd`: Jensen-Shannon divergence similarity
      - `--mod l2s`: Euclidean distance similarity
    - `--method`: specify the training scheme, default: `4`
      - `--method 0`: training use cross-entropy
      - `--method 1`: training use Bacis DSS scheme
      - `--method 2`: training use MinSim DSS scheme
      - `--method 4`: training use MinSim+ DSS scheme
    - `--lam`: the weight of $\mathcal{L}\_{sim}$, default: `30`
    - `--alpha`: the weight of $\mathcal{L}\_{max}$ or $\mathcal{L}\_{neg}$, default: `0.1`
