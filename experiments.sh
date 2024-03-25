###############
# Nodes Lists #
###############

### Node1 Base: Res18_HLB GM

# Purify
python3 purify.py --remote_user 'sunaybhat' --ebm_model 'EBMSNGAN32' --ebm_name 'tinyimagenet_ep780_nf256' --ebm_nf 256 --ebm_lang_steps 150 --diff_model None --dataset 'tinyimagenet';
python3 purify.py --remote_user 'sunaybhat' --ebm_model 'EBMSNGAN32' --ebm_name 'tinyimagenet_ep780_nf256' --ebm_nf 256 --ebm_lang_steps 150 --diff_model None --dataset 'tinyimagenet' --poison_type 'GradientMatching';

# Train
python3 train_classifier.py --remote_user 'sunaybhat' --config_override R18_HLB --data_key 'EBMSNGAN32[tinyimagenet_ep780_nf256_nf256]_150Steps_T0.0001' --dataset 'tinyimagenet' --poison_type 'GradientMatching';


## Node 5 Narc 0-10%
# Create Narcissus Poisons
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --num_images_narcissus 0,500,1000,2500,5000 --poison_type 'Narcissus' --num_proc 8;
# create Purified Poisons
python3 purify.py --remote_user 'sunaybhat' --diff_model None --poison_type 'Narcissus' --num_images_narcissus 500,1000,2500,5000 --num_proc 8;
# Puriifeid baseline
python3 purify.py --remote_user 'sunaybhat' --diff_model None;

# Train Narcissus Poisons
for i in 500 1000 2500 5000; do
    python3 train_classifier.py --remote_user 'sunaybhat' --config_override R18_HLB --num_images_narcissus $i --poison_type 'Narcissus';
    python3 train_classifier.py --remote_user 'sunaybhat' --config_override R18_HLB --num_images_narcissus $i --poison_type 'Narcissus' --data_key 'EBMSNGAN32[ebm_cifar10_45k]_150Steps_T0.0001';
done

# No Posions
python3 train_classifier.py --remote_user 'sunaybhat' --config_override R18_HLB --num_images_narcissus $i;
python3 train_classifier.py --remote_user 'sunaybhat' --config_override R18_HLB --num_images_narcissus $i --data_key 'EBMSNGAN32[ebm_cifar10_45k]_150Steps_T0.0001';

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
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --poison_type 'GradientMatching' --dataset 'tinyimagenet';

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

# Copy Poisons/Models Up to Node
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/Research/data_EBM_Defense/* sunaybhat@node1_Base:/home/sunaybhat/data/

# Copy EBM Data down to local
rsync -av "sunaybhat@node7:/home/sunaybhat/models/*" /Users/sunaybhat/Documents/GitHub/models/
rsync -av "sunaybhat@node8:/home/sunaybhat/models/*" /Users/sunaybhat/Documents/GitHub/models/


# Copy Cifar10 Split Data
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


