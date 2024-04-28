###############
# Nodes Lists #
###############

### Node 3: NTG Attack
(
# Medium 500 Steps Model
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[500]_nf[M]_ep[125]' --unet_channels M --diff_T 200,150,125,100,75,50,25,10  --poison_type 'NeuralTangent' --num_proc 8;
)

### Node 1,4,5 Narc Eps 16
(
# Large 1000 Steps Model 
# python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[1000]_nf[L]_ep[200]' --unet_channels L --diff_T 200,150,125,100,75,50,25,10  --num_proc 8;
# python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[1000]_nf[L]_ep[200]' --unet_channels L --diff_T 200,150,125,100,75,50,25,10  --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
# Medium 1000 Steps Model
# python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[1000]_nf[M]_ep[150]' --unet_channels M --diff_T 200,150,125,100,75,50,25,10  --num_proc 8;
# python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[1000]_nf[M]_ep[150]' --unet_channels M --diff_T 200,150,125,100,75,50,25,10  --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
# Large 750 Steps Model
# python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[750]_nf[L]_ep[175]' --unet_channels L --diff_T 200,150,125,100,75,50,25,10  --num_proc 8;
# python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[750]_nf[L]_ep[175]' --unet_channels L --diff_T 200,150,125,100,75,50,25,10  --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
# Medium 750 Steps Model
# python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[750]_nf[M]_ep[150]' --unet_channels M --diff_T 200,150,125,100,75,50,25,10  --num_proc 8;
# python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[750]_nf[M]_ep[150]' --unet_channels M --diff_T 200,150,125,100,75,50,25,10  --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
# Large 500 Steps Model
# python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[500]_nf[L]_ep[175]' --unet_channels L --diff_T 200,150,125,100,75,50,25,10  --num_proc 8;
# python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[500]_nf[L]_ep[175]' --unet_channels L --diff_T 200,150,125,100,75,50,25,10  --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
## Large 250 Steps Model
# python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[250]_nf[L]_ep[150]' --unet_channels L --diff_T 200,150,125,100,75,50,25,10  --num_proc 8;
# python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[250]_nf[L]_ep[150]' --unet_channels L --diff_T 200,150,125,100,75,50,25,10  --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
# Medium 250 Steps Model
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[250]_nf[M]_ep[150]' --unet_channels M --diff_T 200,150,125,100,75,50,25,10  --num_proc 8;
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[250]_nf[M]_ep[150]' --unet_channels M --diff_T 200,150,125,100,75,50,25,10  --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;

)

## Node4:
(
# Large 150 Steps Model
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[150]_nf[L]_ep[100]' --unet_channels L --diff_T 200,150,125,100,75,50,25,10  --num_proc 8;
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[150]_nf[L]_ep[100]' --unet_channels L --diff_T 200,150,125,100,75,50,25,10  --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
# Medium 500 Steps Model
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[500]_nf[M]_ep[125]' --unet_channels M --diff_T 200,150,125,100,75,50,25,10  --num_proc 8;
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[500]_nf[M]_ep[125]' --unet_channels M --diff_T 200,150,125,100,75,50,25,10  --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
)

### Node1:
(
# Medium 150 Steps Model
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[150]_nf[M]_ep[75]' --unet_channels M --diff_T 200,150,125,100,75,50,25,10  --num_proc 8;
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[150]_nf[M]_ep[75]' --unet_channels M --diff_T 200,150,125,100,75,50,25,10  --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
)

####################
# Purificatiion #
####################


### Node 3:
for i in 200 150 125 100 75 50 25 10
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[cifar10_DDPM[500]_nf[M]_ep[125]]_T[$i]" --poison_type 'NeuralTangent';
done

### Node 5 Narc Eps 16
for i in 200 150 125 100 75 50 25 10
do 
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[cifar10_DDPM[150]_nf[(64, 64, 128, 128, 256, 256)]]_T[$i]" --poison_type 'Narcissus' --noise_eps_narcissus 16;
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[cinic10_imagenet_DDPM[150]_nf[(32, 32, 64, 64, 128, 128)]]_T[$i]" --poison_type 'Narcissus' --noise_eps_narcissus 16;
done

 
###############
# Experiments #
###############

############################
# Setup Node and Copy Data #
############################

# Copy Models up to Node
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node1:/home/sunaybhat/data/PureGen_Models/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node3:/home/sunaybhat/data/PureGen_Models/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node4:/home/sunaybhat/data/PureGen_Models/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node5:/home/sunaybhat/data/PureGen_Models/;

# Copy Poisons Up to Node
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/Poisons/* sunaybhat@node1:/home/sunaybhat/data/Poisons/;

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

(

)

