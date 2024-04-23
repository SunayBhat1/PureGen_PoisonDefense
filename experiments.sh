###############
# Nodes Lists #
###############

# Check Jpeg integration
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --jpeg_compression 75;
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --jpeg_compression 75 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --jpeg_compression 75 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --jpeg_compression 75 --poison_type 'Narcissus' --noise_eps_narcissus 16;


### Node3: HF Diffusion Baselines

for i in 150 200
do
    python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model 'HF_DDPM' --diff_T $i;
    python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model 'HF_DDPM' --diff_T $i --poison_type 'Narcissus' --noise_eps_narcissus 16;
    python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model 'HF_DDPM' --diff_T $i --poison_type 'Narcissus';
done

# Train Classifiers
(
# Narc eps 16
python3 train_classifier.py --remote_user 'sunaybhat' --noise_eps_narcissus 16 --data_key 'EBM[cinic10_imagenet_ep120_nf32]_Steps[150]_T[0.0001]';
python3 train_classifier.py --remote_user 'sunaybhat' --noise_eps_narcissus 16 --data_key 'EBMSNGAN32[cinic10_ep585_nf192]_Steps[150]_T[0.0001]';
python3 train_classifier.py --remote_user 'sunaybhat' --noise_eps_narcissus 16 --data_key 'HF_DDPM[google_ddpm-cifar10-32]_T[10]';
python3 train_classifier.py --remote_user 'sunaybhat' --noise_eps_narcissus 16 --data_key 'HF_DDPM[google_ddpm-cifar10-32]_T[25]';
python3 train_classifier.py --remote_user 'sunaybhat' --noise_eps_narcissus 16 --data_key 'HF_DDPM[google_ddpm-cifar10-32]_T[50]';
python3 train_classifier.py --remote_user 'sunaybhat' --noise_eps_narcissus 16 --data_key 'HF_DDPM[google_ddpm-cifar10-32]_T[75]';
python3 train_classifier.py --remote_user 'sunaybhat' --noise_eps_narcissus 16 --data_key 'HF_DDPM[google_ddpm-cifar10-32]_T[100]';
python3 train_classifier.py --remote_user 'sunaybhat' --noise_eps_narcissus 16 --data_key 'HF_DDPM[google_ddpm-cifar10-32]_T[125]';
)

####################
# Purificatiion #
####################

# Base Dataset
python3 purify.py --remote_user 'sunaybhat'; # --ebm_model None --diff_model None ## To remove either model

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
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/Poisons/* sunaybhat@node3:/home/sunaybhat/data/Poisons/;
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

(

)

