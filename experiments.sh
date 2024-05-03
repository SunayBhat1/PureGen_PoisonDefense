###############
# Nodes Lists #
###############

### Node 1Purify Poisoned EBM
(
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_nf[32]_NS[num=5000_size=32_eps=8]' --num_proc 8 --ebm_lang_steps 2000,1000,750,150;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_nf[32]_NS[num=5000_size=32_eps=8]' --ebm_lang_steps 150 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_nf[32]_NS[num=5000_size=32_eps=8]' --num_proc 8 --ebm_lang_steps 2000,1000,750 --poison_type 'Narcissus' --noise_eps_narcissus 16;
)

# Train classifer
python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[cifar10_nf[32]_NS[num=5000_size=32_eps=8]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16;
python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[cifar10_nf[32]_NS[num=5000_size=32_eps=8]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus';





### Node4: Purify DM
(
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[250]_nf[L]_NS[num=5000_size=32_eps=8]' --num_proc 8 --diff_T 150,125,100,75;
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[250]_nf[L]_NS[num=5000_size=32_eps=8]' --num_proc 8 --diff_T 150,125,100,75 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[250]_nf[L]_NS[num=5000_size=32_eps=8]' --num_proc 8 --diff_T 150,125,100,75 --poison_type 'Narcissus' --noise_eps_narcissus 16;
)


############################
# Setup Node and Copy Data #
############################

# Copy Models up to Node
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node1:/home/sunaybhat/data/PureGen_Models/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node2:/home/sunaybhat/data/PureGen_Models/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node3:/home/sunaybhat/data/PureGen_Models/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node4:/home/sunaybhat/data/PureGen_Models/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node5:/home/sunaybhat/data/PureGen_Models/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node7:/home/sunaybhat/data/PureGen_Models/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node8:/home/sunaybhat/data/PureGen_Models/;

# Copy Poisons Up to Node
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/Poisons/* sunaybhat@node8:/home/sunaybhat/data/Poisons/;

# Copy Cifar10 Split Data Up to Node
scp /Users/sunaybhat/Documents/GitHub/Research/data/CIFAR10_TRAIN_Splith sunaybhat@Calt3_dani:/home/sunaybhat/data/;


(
# Clone 
mkdir data;
git clone https://github.com/SunayBhat1/PureGen_PoisonDefense
# Create a data dir
pip install tqdm;
pip install pandas;
pip install diffusers;
pip install --upgrade diffusers transformers accelerate scipy ftfy safetensors;
pip install pytorch-fid;
# Submodlib install
git clone https://github.com/decile-team/submodlib.git;
cd submodlib;
pip install .;
cd ..;
rm -rf submodlib;
)
