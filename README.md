# [PureGEN: Universal Data Purification for Train-Time Poison Defense via Generative Model Dynamics](https://arxiv.org/abs/2405.18627)

See our [paper on arXiv](https://arxiv.org/abs/2405.18627), (as well as [PureEBM](https://arxiv.org/abs/2405.19376), *a subset work with more details on EBM poison defense*) 

## Introduction

Train-time data poisoning attacks threaten machine learning models by introducing adversarial examples during training, leading to misclassification. Current defense methods often reduce generalization performance, are attack-specific, and impose significant training overhead. To address this, we introduce a set of universal data purification methods using a stochastic transform, $\Psi(x)$, realized via iterative Langevin dynamics of Energy-Based Models (EBMs), Denoising Diffusion Probabilistic Models (DDPMs), or both. These approaches purify poisoned data with minimal impact on classifier generalization. Our specially trained EBMs and DDPMs provide state-of-the-art defense against various attacks (including Narcissus, Bullseye Polytope, Gradient Matching) on CIFAR-10, Tiny-ImageNet, and CINIC-10, without needing attack or classifier-specific information. We discuss performance trade-offs and show that our methods remain highly effective even with poisoned or distributionally shifted generative model training data.

![PureGen Pipeline](imgs/pgen_pipeline.png)

### Key Contributions

* A set of state-of-the-art (SoTA) stochastic preprocessing defenses $\Psi(x)$ against adversarial poisons using MCMC dynamics of EBMs and DDPMs trained specifically for purification named PureGen-EBM and PureGen-DDPM and used in combination with techniques PureGen-Naive PureGen-Reps, and PureGen-Filt.
* Experimental results showing the broad application of $\Psi(x)$ with minimal tuning and no prior knowledge needed of the poison type and classification model
* Results showing SoTA performance can be maintained even when PureGen models’ training data includes poisons or is from a significantly different distribution than the classifier/attacked train data distribution.

### Generative Model Dynamics Pushes Poisoned Images Into the Clean Data Manifold

![PureGen Pipeline](imgs/energy_dists.png)


## Installation and Setup (TPU Node Only)

To install and run this project, follow these steps (currently for TPU with `tpu-vm-pt-2.0` software version)

1. Clone the repository: 
    ```bash
    git clone https://github.com/SunayBhat1/PureGen_PoisonDefense
    ```
2. Make a Data Directory (modify `--data_dir` arg if different)
    ```bash
    mkdir data
    ``` 
3. Navigate to the project directory: 
    ```bash
    cd PureGen_PoisonDefense
    ```
4. Install the required packages: 
    - TPU: 
        ```bash
        pip install -r requirements_TPU.txt
        ```
    - GPU (*TBD*)
6. Download pretrained transfer models and store under /<data_dir>/PureGen_Models/
    - [Pretrained Transfer Models](https://drive.google.com/drive/folders/1FEhrorad9oREboCevwidrCRduXL8pM26?usp=sharing), store in `/<data_dir>/PureGen_Models/transfer_models/` *if running fine-tune or linear transfer poison scenarios* (credit BullseyePoison see below)
7. Datasets need to be uploaded to the `data_dir` above
    - CIFAR-10 will auto-download
    - [CIFAR-10 transfer split](https://drive.google.com/file/d/1bU8mz-MuJN2z7ZZjrhSGBmmiDlJpw3GM/view?usp=sharing) (credit BullseyePoison see below)
    - [tiny-imagenet-200](https://www.kaggle.com/datasets/nikhilshingadiya/tinyimagenet200)
    - [CINIC-10](https://datashare.ed.ac.uk/handle/10283/3192)

## Quick Start

```bash
# Set your remote user (for TPU)
remote_user='your_username';
```

### 1. Train PureGen Models

```bash
python3 scripts/train_EBM.py --remote_user $remote_user; # Train EBM
python3 scripts/train_DM.py --remote_user $remote_user; # Train DDPM
```

### 2. Purification

Purified data will be saved in `/<args.data_dir>/PureGen_PurifiedData/<args.dataset>` under base dataset or poison path and with a purification `data_key` name

```bash
# Baseline (no defense)
python3 scripts/purify.py --remote_user $remote_user;
python3 scripts/purify.py --remote_user $remote_user --poison_type 'Narcissus';
# PureGen-EBM defense
python3 scripts/purify.py --remote_user $remote_user --ebm_model;
python3 scripts/purify.py --remote_user $remote_user --ebm_model --poison_type 'Narcissus';
# PureGen-DDPM defense  
python3 scripts/purify.py --remote_user $remote_user --diff_model;
python3 scripts/purify.py --remote_user $remote_user --diff_model --poison_type 'Narcissus';
```

### 3. Train Classifiers with Poisoned and Purified Data
```bash
python3 scripts/train_classifier.py --remote_user $remote_user;
python3 scripts/train_classifier.py --remote_user $remote_user --data_key "EBM[Steps[150]]";
```

## Additional Configurations

**Poison Modes**:  
    - From Scratch (Default): ```--poison_mode 'from_scratch'```  
    - Transfer Base Dataset (CIFAR-10 only): ```--poison_type 'TransferBase'```  
    - Linear Transfer: ```--poison_mode 'linear_transfer'```  
    - Fine-Tune Transfer: ```--poison_mode 'fine_tune_transfer'```  
*Note that both transfer modes use same transfer dataset provided by Bullseye Poison Authors*  

**Poison Types**:  
    - Narcissus (Default, Triggered): ```--poison_type 'Narcissus'```  
    - Gradient Matching (Triggerless): ```--poison_type 'GradientMatching'```  
    - Bullseye Polytope (Triggerless): ```--poison_type 'BullseyePolytope'```  

**Classifier Models**  
    - HyperLight Benchmark (Default)  
        -- Small: `--config_overrides 'HLB_SMALL'`  
        -- Medium: `--config_overrides 'HLB_MEDIUM'`  
        -- Large: `--config_overrides 'HLB_LARGE'`  
    - ResNet18 HLB: `--config_overrides 'R18_HLB'`  
    - MobileNetV2: `--config_overrides 'MOBILE_NET'`  
    - DenseNet121: `--config_overrides 'DENSE_NET'`  

**Training Scenarios**  
    - Linear Transfer: `--config_overrides 'LINEAR_TRANSFER'`  
    - Fine-Tune Transfer: `--config_overrides 'FINE_TUNE'`  
    - 80 Epoch From Scratch: `--config_overrides '80_EPOCH'`  

**Baseline Defenses**  
    - JPEG Compression:  
        -- Add to `purify.py` `--jpeg_compression <Compression Ratio>` to any of the purify runs (typically 75 or 85 for ratio)  
        -- Add to `train_classifier.py` `--data_key 'JEPG[<Compression Ratio>]'` to any of the train classifier runs (same ratio as purify)   
    - Friendly Noise:  
        -- Add to `train_classifier.py` `--baseline_defense 'Friendly'` to any of the train classifier runs  
    - Epic:  
        -- Install Submodlib:  
            ```bash
            git clone https://github.com/decile-team/submodlib.git;
            cd submodlib;
            pip install .;
            cd ..;
            rm -rf submodlib;
            ```    
        -- Add to `train_classifier.py` `--baseline_defense 'Epic'` to any of the train classifier runs   


Additional configuations and parameters can be found in the `Configs/config.ini` file.  
*Note: `config_overrides` arguments can be chained if desired, they execute in order*


## Analyze Results

See the `Results.ipynb` to examples of how to read and parse the results csv's that are saved. 

## License

Shield: [![CC BY-ND 4.0][cc-by-nd-shield]][cc-by-nd]

This work is licensed under a
[Creative Commons Attribution-NoDerivs 4.0 International License][cc-by-nd].

[![CC BY-ND 4.0][cc-by-nd-image]][cc-by-nd]

[cc-by-nd]: https://creativecommons.org/licenses/by-nd/4.0/
[cc-by-nd-image]: https://licensebuttons.net/l/by-nd/4.0/88x31.png
[cc-by-nd-shield]: https://img.shields.io/badge/License-CC%20BY--ND%204.0-lightgrey.svg

## References

This project was built using the following open-source repositories:

- [BullseyePoison](https://github.com/ucsb-seclab/BullseyePoison): Bullseye Polytope poisons, transfer models, and transfer dataset.
- [EPIC](https://github.com/YuYang0901/EPIC): Epic defense code was added to `train_classifier.py` and `utils_baslines.py`.
- [Friendly Noise](https://github.com/tianyu139/friendly-noise): Friendly noise defense code was added to `train_classifier.py` and `utils_baslines.py`.
- [Poisoning Benchmarks](https://github.com/aks2203/poisoning-benchmark/tree/master): BP Benchmarks and dataloaders users for white-box R18 attack
- [HyperLight Benchmark](https://github.com/tysam-code/hlb-CIFAR10): Super Fast convergence on CIFAR-10 models

We would like to thank the authors of these repositories for their contributions to the open-source community.

## Acknowledgements
This work is supported with Cloud TPUs from [Google’s Tensorflow Research Cloud (TFRC)](https://sites.research.google/trc/about/).

## Contact

If you have any questions or feedback, please contact [Sunay Bhat](mailto:sunaybhat1@ucla.edu).
