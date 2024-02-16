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
### Calt1
### Calt2
### Calt3_dani
### Calt4_david
### Calt5_rez

### Node1 Base: Testing


####################
# Core Experiments #
####################

### Defenses
--defense 'Epic' --epic_subset_size 0.1 --epic_drop_after 10
--defense 'Epic' --epic_subset_size 0.2 --epic_drop_after 20
--defense 'Epic' --epic_subset_size 0.3 --epic_drop_after 30
--defense 'Friendly' 
--defense 'Friendly' --friendly_noise_type 'friendly' 'gaussian'
--defense 'EBM'

### From Scratch
# Gradient Matching
python3 run.py --remote_user 'sunaybhat' --poison_type 'Gradient_Matching' --defense 'None';
# Narcissus
python3 run.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --defense 'None';

### Tranfer Learning
# Bullseye Polytope Linear-Transfera
python3 run.py --remote_user 'sunaybhat' --poison_type 'BullseyePolytope' --defense 'None' --poison_mode 'transfer';
# Bullseye Polytope Fine-Tune
python3 run.py --remote_user 'sunaybhat' --poison_type 'BullseyePolytope' --defense 'None' --poison_mode 'transfer' --fine_tune;
# Narcissus  Fine-Tune
python3 run.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --defense 'None' --poison_mode 'transfer' --fine_tune --config_override 'NARC_FINE_TUNE_50_50'; # ['NARC_FINE_TUNE_50_200'],['NARC_FINE_TUNE_20_200']
# Bullseye Polytope Benchmark Linear-Transfer
python3 run.py --remote_user 'sunaybhat' --poison_type 'BullseyePolytope_Bench' --defense 'None' --poison_mode 'transfer' --config_override 'TRANSFER_BENCH';

###############
# Experiments #
###############

### Node 9 timing 

for i in 0 8
do
    python3 run.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --defense 'Diff' --start_target_index $i;
    python3 run.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --defense 'EBM' --start_target_index $i;
    python3 run.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --defense 'EBM_Diff' --start_target_index $i;
    python3 run.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --defense 'None' --start_target_index $i;
done

############################
# Setup Node and Copy Data #
############################

# Copy Poisons/Models
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/Research/data_EBM_Defense/* sunaybhat@node1_Base:/home/sunaybhat/data/



# Copy Cifar10 Split Data
scp /Users/sunaybhat/Documents/GitHub/Research/data/CIFAR10_TRAIN_Split.pth sunaybhat@Calt3_dani:/home/sunaybhat/data/;

# Delete Results
ssh sunaybhat@node1_Base 'rm -rf /home/sunaybhat/results_EBM_Defense';


(
# Clone 
mkdir data;
git clone https://github.com/SunayBhat1/EBM_Poison_Defense
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


