###############
# Nodes Lists #
###############

### Node1_Base (Dev)
### Node10 (Dev)
### Node1
### Node2
### Node3
### Node4
### Node5
### Node6
### Node7
### Node8
### Node9

### Node1 Base: Testing HLB Integrations

python3 train_classifier.py --remote_user 'sunaybhat' --config_override R18_HLB --no_poison --num_proc 1 --dataset 'stl10';


python3 train_classifier.py --remote_user 'sunaybhat' --config_override ResNet18 --dataset 'stl10' --no_poison --num_proc 1;


# Clean HLB Runs
python3 train_classifier.py --remote_user 'sunaybhat' --no_poison --num_proc 1;
python3 train_classifier.py --remote_user 'sunaybhat' --config_override HLB_MED --no_poison --num_proc 1;
python3 train_classifier.py --remote_user 'sunaybhat' --config_override HLB_LARGE --no_poison --num_proc 1;
python3 train_classifier.py --remote_user 'sunaybhat' --config_override R18_HLB --no_poison --num_proc 1;

# Poisoned HLB Runs Baseline
(
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --poison_type 'Narcissus';

python3 train_classifier.py --remote_user 'sunaybhat' --data_key 
python3 train_classifier.py --remote_user 'sunaybhat' --config_override HLB_MED;
python3 train_classifier.py --remote_user 'sunaybhat' --config_override HLB_LARGE;
python3 train_classifier.py --remote_user 'sunaybhat' --config_override R18_HLB;

python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --poison_type 'Narcissus' --noise_eps_narcissus 16;

python3 train_classifier.py --remote_user 'sunaybhat' --noise_eps_narcissus 16;
python3 train_classifier.py --remote_user 'sunaybhat' --config_override HLB_MED --noise_eps_narcissus 16;
python3 train_classifier.py --remote_user 'sunaybhat' --config_override HLB_LARGE --noise_eps_narcissus 16;
python3 train_classifier.py --remote_user 'sunaybhat' --config_override R18_HLB --noise_eps_narcissus 16;
)

## STL 10
python3 train_classifier.py --remote_user 'sunaybhat' --config_override ResNet18 --dataset 'stl10' --no_poison --num_proc 1;
python3 train_classifier.py --remote_user 'sunaybhat' --config_override ResNet18 --dataset 'stl10' --poison_type 'Narcissus';


# ResNet18 Testing
python3 train_classifier.py --remote_user 'sunaybhat' --config_override ResNet18 --no_poison --num_proc 1;
python3 train_classifier.py --remote_user 'sunaybhat' --config_override ResNet18;


### Node 8: Train small EBMS CINIC-10 
python3 EBM/train_EBM.py --dataset 'cincic10_imagenet_subset';
python3 EBM/train_EBM.py --dataset 'cincic10_imagenet_subset' --lr 1e-3;
python3 EBM/train_EBM.py --dataset 'cincic10_imagenet_subset' --epochs 100 --lr_decay_milestones 35 65 85; 




### Node 9 Train STL R18 Classifier
python3 run.py --dataset 'stl10'

####################
# Purificatiion #
####################

python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --dataset 'stl10';
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --poison_type 'Narcissus' --dataset 'stl10' --num_images_narcissus 100;
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --poison_type 'Narcissus' --noise_eps_narcissus 16 --dataset 'stl10' --num_images_narcissus 100;


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
rsync -av "sunaybhat@node6:/home/sunaybhat/models/ebms/*" /Users/sunaybhat/Documents/GitHub/models/ebm/

# Copy Cifar10 Split Data
scp /Users/sunaybhat/Documents/GitHub/Research/data/CIFAR10_TRAIN_Split.pth sunaybhat@Calt3_dani:/home/sunaybhat/data/;


(
# Clone 
mkdir data;
git clone https://github.com/SunayBhat1/PureDefense
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


