###############
# Nodes Lists #
###############

### Node1: R18 HLB Baselines Friends
(
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --config_overrides 'ResNet18' --baseline_defense 'Friendly' --config_overrides 'NARC_10';
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'ResNet18' --baseline_defense 'Friendly' --config_overrides 'NARC_10';
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --config_overrides 'ResNet18' --baseline_defense 'ResNet18' --friendly_noise_type 'friendly' 'gaussian' --config_overrides 'NARC_10';
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB' --baseline_defense 'Friendly' --friendly_noise_type 'friendly' 'gaussian' --config_overrides 'NARC_10';
)

### Node2: R18 HLB Transfer Baselines
(
# Purify Transfer Base
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --poison_type 'TransferBase';
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --poison_type 'TransferBase' --num_proc 8 --jpeg_compression 25,50,75,85;
# Purify Narc Transfer
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --poison_type 'Narcissus' --poison_mode 'fine_tune_transfer' --num_images_narcissus 20;
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --poison_type 'Narcissus' --poison_mode 'fine_tune_transfer' --num_images_narcissus 20 --noise_eps_narcissus 16
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --poison_type 'Narcissus' --poison_mode 'fine_tune_transfer' --num_images_narcissus 200;
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --poison_type 'Narcissus' --poison_mode 'fine_tune_transfer' --num_images_narcissus 200 --noise_eps_narcissus 16;
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --poison_type 'Narcissus' --poison_mode 'fine_tune_transfer' --num_images_narcissus 20 --num_proc 8 --jpeg_compression 25,50,75,85;
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --poison_type 'Narcissus' --poison_mode 'fine_tune_transfer' --num_images_narcissus 20 --noise_eps_narcissus 16 --num_proc 8 --jpeg_compression 25,50,75,85;
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --poison_type 'Narcissus' --poison_mode 'fine_tune_transfer' --num_images_narcissus 200 --num_proc 8 --jpeg_compression 25,50,75,85;
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_model None --poison_type 'Narcissus' --poison_mode 'fine_tune_transfer' --num_images_narcissus 200 --noise_eps_narcissus 16 --num_proc 8 --jpeg_compression 25,50,75,85;
)

(
# Train Clf No Defense
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --poison_mode 'fine_tune_transfer' --config_overrides 'R18_HLB' 'NARC_1_TRANSFER';
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB' 'NARC_1_TRANSFER';
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --poison_mode 'fine_tune_transfer' --config_overrides 'R18_HLB' 'NARC_10_TRANSFER';
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB' 'NARC_10_TRANSFER';

# Train Clf JPEG
for i in 25 50 75 85; do
    python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --data_key "JPEG[$i]" --poison_mode 'fine_tune_transfer' --config_overrides 'R18_HLB' 'NARC_1_TRANSFER';
    python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --data_key "JPEG[$i]" --noise_eps_narcissus 16 --config_overrides 'R18_HLB' 'NARC_1_TRANSFER';
    python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --data_key "JPEG[$i]" --poison_mode 'fine_tune_transfer' --config_overrides 'R18_HLB' 'NARC_10_TRANSFER';
    python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --data_key "JPEG[$i]" --noise_eps_narcissus 16 --config_overrides 'R18_HLB' 'NARC_10_TRANSFER';
done;
)


### Node3: R18 HLB Baselines Epic
(
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --config_overrides 'ResNet18' --baseline_defense 'Epic' --epic_subset_size 0.1;
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'ResNet18' --baseline_defense 'Epic' --epic_subset_size 0.1;
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --config_overrides 'ResNet18' --baseline_defense 'Epic' --epic_subset_size 0.2;
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'ResNet18' --baseline_defense 'Epic' --epic_subset_size 0.2;
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --config_overrides 'ResNet18' --baseline_defense 'Epic' --epic_subset_size 0.3;
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'ResNet18' --baseline_defense 'Epic' --epic_subset_size 0.3;
)

### Node4: R18 HLB EBM/DM
(
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_lang_steps 150 --poison_type 'Narcissus' --num_images_narcissus 5000;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_lang_steps 2000,1000,750 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 --num_images_narcissus 5000;

# Train Classifier
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[150]_T[0.0001]" --config_overrides 'R18_HLB' --num_images_narcissus 5000;
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --data_key "DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[75]" --config_overrides 'R18_HLB' --num_images_narcissus 5000;
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[750]_T[0.0001]" --noise_eps_narcissus 16 --config_overrides 'R18_HLB' --num_images_narcissus 5000;
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[1000]_T[0.0001]" --noise_eps_narcissus 16 --config_overrides 'R18_HLB' --num_images_narcissus 5000;
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[2000]_T[0.0001]" --noise_eps_narcissus 16 --config_overrides 'R18_HLB' --num_images_narcissus 5000;
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --data_key "DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[125]" --noise_eps_narcissus 16 --config_overrides 'R18_HLB' --num_images_narcissus 5000;
)


### Node5: EBM Filter
(
python3 purify.py --remote_user 'sunaybhat' --ebm_lang_steps 250,150,150 --diff_T 125,125,75 --num_proc 8;
python3 purify.py --remote_user 'sunaybhat' --ebm_lang_steps 250,150 --diff_T 125 --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16;
python3 purify.py --remote_user 'sunaybhat' --ebm_lang_steps 150 --diff_T 75 --poison_type 'Narcissus';

# Train Classifier
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[150]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[75]" --config_overrides 'R18_HLB';
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[150]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[125]" --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[250]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[125]" --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
)


############
# Run Info #
############

### EBm Filter
--ebm_filter $j

### Narcissus
--poison_type 'NeuralTangent'
--poison_type 'NeuralTangent' --noise_eps_narcissus 16

### Node3: NTGA Reps
(
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'NeuralTangent' --config_overrides 'HLB_LARGE' \
    --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[50]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[25]_reps6"
)


# python3 purify.py --remote_user 'sunaybhat' --purify_reps 2 --num_proc 8 \
#     --ebm_lang_steps 50,50,50,50,25,25,25,25 \
#     --diff_T 75,50,25,10,75,50,25,10;
# python3 purify.py --remote_user 'sunaybhat' --purify_reps 2 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 \
#     --ebm_lang_steps 50,50,50,50,25,25,25,25 \
#     --diff_T 75,50,25,10,75,50,25,10;

############################
# Setup Node and Copy Data #
############################

rsync -av sunaybhat@node1:/home/sunaybhat/data/PureGen_PoisonDefense/* /Users/sunaybhat/Documents/GitHub/data/PureGen_PoisonDefense/;
rsync -av --exclude='.DS_Store' sunaybhat@node1:/home/sunaybhat/data/PureGen_Models/ /Users/sunaybhat/Documents/GitHub/data/PureGen_PoisonDefense/;

# Copy Models up to Node
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node1:/home/sunaybhat/data/PureGen_Models/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node2:/home/sunaybhat/data/PureGen_Models/;
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
