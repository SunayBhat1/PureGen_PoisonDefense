###############
# Nodes Lists #
###############

    # if rank == 0:
    #     args.diff_name = 'cinic10_imagenet_EBMGuided_F10_nf64_modeEps_mcmc100_ep80'
    #     args.diff_ebm_guided = True
    #     args.diff_mode = 'Eps'
    # elif rank == 1:
    #     args.diff_name = 'cinic10_imagenet_EBMGuided_F10_nf64_modeX0_mcmc100_ep80'
    #     args.diff_ebm_guided = True
    #     args.diff_mode = 'X0'
    # elif rank == 2:
    #     args.diff_name = 'cinic10_imagenet_nf64_modeEps_mcmc100_ep90'
    #     args.diff_ebm_guided = False
    #     args.diff_mode = 'Eps'
    # elif rank == 3:
    #     args.diff_name = 'cinic10_imagenet_nf64_modeX0_mcmc100_ep80'
    #     args.diff_ebm_guided = False
    #     args.diff_mode = 'X0'



### Node6:
# Compare Diffusion eps 8
(
# python3 train_classifier.py --remote_user 'sunaybhat';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key 'EBM[cinic10_imagenet_ep120_nf32]_Steps[1000]_T[0.0001]';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key 'EBM[cinic10_imagenet_ep120_nf32]_Steps[1000]_T[0.0001]_DM_UNET_SMALL[cinic10_imagenet_nf64_modeX0_mcmc100_ep80]_Steps[[0, 1, 1]]';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key 'EBM[cinic10_imagenet_ep120_nf32]_Steps[1000]_T[0.0001]_DM_UNET_SMALL[cinic10_imagenet_nf64_modeEps_mcmc100_ep90]_Steps[[0, 1, 1]]';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key 'EBM[cinic10_imagenet_ep120_nf32]_Steps[1000]_T[0.0001]_DM_UNET_SMALL[cinic10_imagenet_EBMGuided_F10_nf64_modeX0_mcmc100_ep80]_Steps[[0, 1, 1]]';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key 'EBM[cinic10_imagenet_ep120_nf32]_Steps[1000]_T[0.0001]_DM_UNET_SMALL[cinic10_imagenet_EBMGuided_F10_nf64_modeEps_mcmc100_ep80]_Steps[[0, 1, 1]]';
# Compare Diffusion eps 16
# python3 train_classifier.py --remote_user 'sunaybhat' --noise_eps_narcissus 16;
python3 train_classifier.py --remote_user 'sunaybhat' --noise_eps_narcissus 16 --data_key 'EBM[cinic10_imagenet_ep120_nf32]_Steps[1000]_T[0.0001]';
python3 train_classifier.py --remote_user 'sunaybhat' --noise_eps_narcissus 16 --data_key 'EBM[cinic10_imagenet_ep120_nf32]_Steps[1000]_T[0.0001]_DM_UNET_SMALL[cinic10_imagenet_nf64_modeX0_mcmc100_ep80]_Steps[[0, 1, 1]]';
python3 train_classifier.py --remote_user 'sunaybhat' --noise_eps_narcissus 16 --data_key 'EBM[cinic10_imagenet_ep120_nf32]_Steps[1000]_T[0.0001]_DM_UNET_SMALL[cinic10_imagenet_nf64_modeEps_mcmc100_ep90]_Steps[[0, 1, 1]]';
python3 train_classifier.py --remote_user 'sunaybhat' --noise_eps_narcissus 16 --data_key 'EBM[cinic10_imagenet_ep120_nf32]_Steps[1000]_T[0.0001]_DM_UNET_SMALL[cinic10_imagenet_EBMGuided_F10_nf64_modeX0_mcmc100_ep80]_Steps[[0, 1, 1]]';
python3 train_classifier.py --remote_user 'sunaybhat' --noise_eps_narcissus 16 --data_key 'EBM[cinic10_imagenet_ep120_nf32]_Steps[1000]_T[0.0001]_DM_UNET_SMALL[cinic10_imagenet_EBMGuided_F10_nf64_modeEps_mcmc100_ep80]_Steps[[0, 1, 1]]';
)


####################
# Purificatiion #
####################

# Base Dataset
python3 purify.py --remote_user 'sunaybhat';

# Poisons
--poison_type 'Narcissus' # --noise_eps_narcissus 16

 
###############
# Experiments #
###############

############################
# Setup Node and Copy Data #
############################

# Copy Models up to Node
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node6:/home/sunaybhat/data/PureGen_Models/;

# Copy Poisons Up to Node
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/Poisons/* sunaybhat@node9:/home/sunaybhat/data/Poisons/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/NGT/* sunaybhat@node1:/home/sunaybhat/data/NGT/

# Copy EBM Data down to local
rsync -av "sunaybhat@node7:/home/sunaybhat/models/*" /Users/sunaybhat/Documents/GitHub/models/

# Copy Cifar10 Split Data Up to Node
scp /Users/sunaybhat/Documents/GitHub/Research/data/CIFAR10_TRAIN_Split.pth sunaybhat@Calt3_dani:/home/sunaybhat/data/;


(
# Clone 
mkdir data;
git clone https://github.com/SunayBhat1/PureGen_PoisonDefense
# Create a data dir
pip install tqdm;
pip install pandas;
# pip install pytorch-fid;
# Submodlib install
git clone https://github.com/decile-team/submodlib.git;
cd submodlib;
pip install .;
cd ..;
rm -rf submodlib;
)


