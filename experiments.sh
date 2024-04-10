###############
# Nodes Lists #
###############

### Node8:
# Compare EBMs Minnesota
python3 purify.py --remote_user 'sunaybhat' --diff_name 'cifar10_ep180_nf64_EBM[cinic10_imagenet_ep120_nf32]_1';
python3 purify.py --remote_user 'sunaybhat' --diff_name 'cifar10_ep180_nf64_EBM[cinic10_imagenet_ep120_nf32]_2';
python3 purify.py --remote_user 'sunaybhat' --diff_name 'cifar10_ep180_nf64_EBM[cinic10_imagenet_ep120_nf32]_1' --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --diff_name 'cifar10_ep180_nf64_EBM[cinic10_imagenet_ep120_nf32]_2' --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --diff_name 'cifar10_ep180_nf64_EBM[cinic10_imagenet_ep120_nf128]';
python3 purify.py --remote_user 'sunaybhat' --diff_name 'cifar10_ep180_nf64_EBM[cinic10_imagenet_ep120_nf128]' --poison_type 'Narcissus';


python3 train_classifier.py --remote_user 'sunaybhat' --data_key 'EBM[cinic10_imagenet_ep120_nf32]_Steps[150]_T[0.0001]_UNET_SMALL[cifar10_ep180_nf64_EBM[cinic10_imagenet_ep120_nf32]_1]_T[150]';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key 'EBM[cinic10_imagenet_ep120_nf32]_Steps[150]_T[0.0001]_UNET_SMALL[cifar10_ep180_nf64_EBM[cinic10_imagenet_ep120_nf32]_2]_T[150]';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key 'EBM[cinic10_imagenet_ep120_nf32]_Steps[150]_T[0.0001]_UNET_SMALL[cifar10_ep180_nf64_EBM[cinic10_imagenet_ep120_nf96]]_T[150]';

# ### Node 8: Train small EBMS CINIC-10 
# python3 EBM/train_EBM.py --dataset 'cincic10_imagenet_subset' --model 'SuperLightEBM' --num_filters 48 --lr 1e-5 --lr_decay_milestones 25 50 75 100;
# python3 EBM/train_EBM.py --dataset 'cincic10_imagenet_subset' --model 'SuperLightEBM' --batch_size 128 --num_filters 48 --lr 1e-5 --lr_decay_milestones 25 50 75 100;

# ### Node 7: Train small EBMS CINIC-10 
# python3 EBM/train_EBM.py --dataset 'cincic10_imagenet_subset' --model 'LightEBM' --lr 1e-5 --lr_decay_milestones 25 50 75 100;
# python3 EBM/train_EBM.py --dataset 'cincic10_imagenet_subset' --model 'LightEBM' --batch_size 64 --lr 1e-5 --lr_decay_milestones 25 50 75 100;

####################
# Purificatiion #
####################



python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --dataset 'tinyimagenet';
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --p3oison_type 'GradientMatching' --dataset 'tinyimagenet';

python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --dataset 'stl10_64';
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --poison_type 'Narcissus' --dataset 'stl10_64' --num_images_narcissus 100;
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --poison_type 'Narcissus' --noise_eps_narcissus 16 --dataset 'stl10_64' --num_images_narcissus 100;

python3 purify.py --remote_user 'sunaybhat' --ebm_lang_steps 150 --diff_model None;
python3 purify.py --remote_user 'sunaybhat' --ebm_lang_steps 150 --diff_model None --poison_type 'Narcissus';

# Base Dataset
python3 purify.py --remote_user 'sunaybhat';

# Poisons
--poison_type 'Narcissus' # 'Gradient_Matching'

# For multiple nodes use 
--num_proc 8 # And pass in a list to any of these (or multiple): --ebm_lang_steps, --ebm_lang_temp, --diff_train_steps, --diff_purify_steps, --diff_eta, --ebm_name, --diff_name, --ebm_nf, --diff_nf

 
###############
# Experiments #
###############

############################
# Setup Node and Copy Data #
############################

# Copy Models up to Node
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node9:/home/sunaybhat/data/PureGen_Models/;

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
git clone https://github.com/SunayBhat1/PureGen_Defense
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


