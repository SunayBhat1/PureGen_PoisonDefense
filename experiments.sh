###############
# Nodes Lists #
###############

# POOD Narcissus

### Node 1
### Node 5
(
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[250]_nf[L]_ep[150]' --diff_T 50 --poison_type 'Narcissus' --noise_eps_narcissus 16;
python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[cifar10_DDPM[250]_nf[L]_ep[150]]_T[25]" --poison_type 'Narcissus' --noise_eps_narcissus 16;
)

    ## node1

(
### EBMs ###
# Cifar-10
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_model 'EBMSNGAN32' --ebm_name 'cifar10_nf[128]' --ebm_lang_steps 1000,750,500,250,150 --num_proc 8 --ebm_nf 128;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_model 'EBMSNGAN32' --ebm_name 'cifar10_nf[128]' --ebm_lang_steps 1000,750,500,250,150 --num_proc 8 --poison_type 'Narcissus' --ebm_nf 128;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_model 'EBMSNGAN32' --ebm_name 'cifar10_nf[128]' --ebm_lang_steps 1000,750,500,250,150 --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16 --ebm_nf 128;
# Cinic-10
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cinic10_imagenet_nf[32]' --ebm_lang_steps 1000,750,500,250,150 --num_proc 8;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cinic10_imagenet_nf[32]' --ebm_lang_steps 1000,750,500,250,150 --num_proc 8 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cinic10_imagenet_nf[32]' --ebm_lang_steps 1000,750,500,250,150 --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
#Office-Home
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'office_home_nf[32]' --ebm_lang_steps 1000,750,500,250,150 --num_proc 8;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'office_home_nf[32]' --ebm_lang_steps 1000,750,500,250,150 --num_proc 8 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'office_home_nf[32]' --ebm_lang_steps 1000,750,500,250,150 --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
#Flowers-102
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'flowers102_nf[32]' --ebm_lang_steps 1000,750,500,250,150 --num_proc 8;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'flowers102_nf[32]' --ebm_lang_steps 1000,750,500,250,150 --num_proc 8 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'flowers102_nf[32]' --ebm_lang_steps 1000,750,500,250,150 --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
#LFW-People
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'lfw_people_nf[32]' --ebm_lang_steps 1000,750,500,250,150 --num_proc 8;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'lfw_people_nf[32]' --ebm_lang_steps 1000,750,500,250,150 --num_proc 8 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'lfw_people_nf[32]' --ebm_lang_steps 1000,750,500,250,150 --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
#Textures
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'textures_nf[32]' --ebm_lang_steps 1000,750,500,250,150 --num_proc 8;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'textures_nf[32]' --ebm_lang_steps 1000,750,500,250,150 --num_proc 8 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'textures_nf[32]' --ebm_lang_steps 1000,750,500,250,150 --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;

# Train Classifier
for i in 1000 750 500 250 150
do
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cifar10_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cifar10_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16;

    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[cinic10_imagenet_nf[32]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[cinic10_imagenet_nf[32]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16;

    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[office_home_nf[32]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[office_home_nf[32]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16;

    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[flowers102_nf[32]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[flowers102_nf[32]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16;

    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[lfw_people_nf[32]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[lfw_people_nf[32]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16;

    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[textures_nf[32]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[textures_nf[32]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16;
done
)


### Diffusion Models ###
(
# Node 4
# Cifar-10
python3 purify.py --remote_user 'sunaybhat'  --ebm_model None --diff_name 'cifar10_DDPM[250]_nf[L]' --diff_T 150,125,100,75 --num_proc 8;
python3 purify.py --remote_user 'sunaybhat'  --ebm_model None --diff_name 'cifar10_DDPM[250]_nf[L]' --diff_T 150,125,100,75 --num_proc 8 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat'  --ebm_model None --diff_name 'cifar10_DDPM[250]_nf[L]' --diff_T 150,125,100,75 --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
# Cinic-10
python3 purify.py --remote_user 'sunaybhat'  --ebm_model None --diff_name 'cinic10_imagenet_DDPM[250]_nf[L]' --diff_T 150,125,100,75 --num_proc 8;
python3 purify.py --remote_user 'sunaybhat'  --ebm_model None --diff_name 'cinic10_imagenet_DDPM[250]_nf[L]' --diff_T 150,125,100,75 --num_proc 8 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat'  --ebm_model None --diff_name 'cinic10_imagenet_DDPM[250]_nf[L]' --diff_T 150,125,100,75 --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
# Office-Home
python3 purify.py --remote_user 'sunaybhat'  --ebm_model None --diff_name 'office_home_DDPM[250]_nf[L]' --diff_T 150,125,100,75 --num_proc 8;
python3 purify.py --remote_user 'sunaybhat'  --ebm_model None --diff_name 'office_home_DDPM[250]_nf[L]' --diff_T 150,125,100,75 --num_proc 8 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat'  --ebm_model None --diff_name 'office_home_DDPM[250]_nf[L]' --diff_T 150,125,100,75 --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
# Flowers-102
python3 purify.py --remote_user 'sunaybhat'  --ebm_model None --diff_name 'flowers102_DDPM[250]_nf[L]' --diff_T 150,125,100,75 --num_proc 8;
python3 purify.py --remote_user 'sunaybhat'  --ebm_model None --diff_name 'flowers102_DDPM[250]_nf[L]' --diff_T 150,125,100,75 --num_proc 8 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat'  --ebm_model None --diff_name 'flowers102_DDPM[250]_nf[L]' --diff_T 150,125,100,75 --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
# LFW-People
python3 purify.py --remote_user 'sunaybhat'  --ebm_model None --diff_name 'lfw_people_DDPM[250]_nf[L]' --diff_T 150,125,100,75 --num_proc 8;
python3 purify.py --remote_user 'sunaybhat'  --ebm_model None --diff_name 'lfw_people_DDPM[250]_nf[L]' --diff_T 150,125,100,75 --num_proc 8 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat'  --ebm_model None --diff_name 'lfw_people_DDPM[250]_nf[L]' --diff_T 150,125,100,75 --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
# Textures
python3 purify.py --remote_user 'sunaybhat'  --ebm_model None --diff_name 'textures_DDPM[250]_nf[L]' --diff_T 150,125,100,75 --num_proc 8;
python3 purify.py --remote_user 'sunaybhat'  --ebm_model None --diff_name 'textures_DDPM[250]_nf[L]' --diff_T 150,125,100,75 --num_proc 8 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat'  --ebm_model None --diff_name 'textures_DDPM[250]_nf[L]' --diff_T 150,125,100,75 --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
)



### Node 3: OOD Neural Tangent
(
# To Run
for i in 150 125 100 75
do
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[cinic10_imagenet_nf[32]]_Steps[$i]_T[0.0001]" --poison_type 'NeuralTangent';
done
)



############################
# Setup Node and Copy Data #
############################

# Copy Models up to Node
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node1:/home/sunaybhat/data/PureGen_Models/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node3:/home/sunaybhat/data/PureGen_Models/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node4:/home/sunaybhat/data/PureGen_Models/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node7:/home/sunaybhat/data/PureGen_Models/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node8:/home/sunaybhat/data/PureGen_Models/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node9:/home/sunaybhat/data/PureGen_Models/;

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

