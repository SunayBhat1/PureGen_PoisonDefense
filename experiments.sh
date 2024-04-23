###############
# Nodes Lists #
###############

# Check Jpeg integration
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --jpeg_compression 85;
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --jpeg_compression 75 --poison_type 'NeuralTangent';
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --jpeg_compression 75 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --jpeg_compression 75 --poison_type 'Narcissus' --noise_eps_narcissus 16;

for i in 25 50 75 85
do
    python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --jpeg_compression $i;
    python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --jpeg_compression $i --poison_type 'Narcissus';
    python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --jpeg_compression $i --poison_type 'Narcissus' --noise_eps_narcissus 16;
done



####################
# Purificatiion #
####################

# Base Dataset
python3 purify.py --remote_user 'sunaybhat'; # --ebm_model None --diff_model None ## To remove either model

# Poisons
--poison_type 'Narcissus' # --noise_eps_narcissus 16
--poison_type 'NeuralTangent'

 
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

