###############
# Nodes Lists #
###############

### Node1: R18 HLB Baselines Friends
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --config_overrides 'R18_HLB' --baseline_defense 'Friendly';
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB' --baseline_defense 'Friendly';
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --config_overrides 'R18_HLB' --baseline_defense 'Friendly' --friendly_noise_type 'friendly' 'gaussian';
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB' --baseline_defense 'Friendly' --friendly_noise_type 'friendly' 'gaussian';

### Node2: R18 HLB Baselines JPEG Compression
(
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --config_overrides 'R18_HLB' --data_key 'JPEG[25]';
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --config_overrides 'R18_HLB' --data_key 'JPEG[50]';
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --config_overrides 'R18_HLB' --data_key 'JPEG[75]';
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --config_overrides 'R18_HLB' --data_key 'JPEG[85]';
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --config_overrides 'R18_HLB' --data_key 'JPEG[25]' --noise_eps_narcissus 16;
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --config_overrides 'R18_HLB' --data_key 'JPEG[50]' --noise_eps_narcissus 16;
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --config_overrides 'R18_HLB' --data_key 'JPEG[75]' --noise_eps_narcissus 16;
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --config_overrides 'R18_HLB' --data_key 'JPEG[85]' --noise_eps_narcissus 16;
)



### Node3: R18 HLB Baselines Gaussian Noise
(
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --config_overrides 'R18_HLB' --baseline_defense 'Epic' --epic_subset_size 0.1;
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB' --baseline_defense 'Epic' --epic_subset_size 0.1;
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --config_overrides 'R18_HLB' --baseline_defense 'Epic' --epic_subset_size 0.2;
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB' --baseline_defense 'Epic' --epic_subset_size 0.2;
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --config_overrides 'R18_HLB' --baseline_defense 'Epic' --epic_subset_size 0.3;
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB' --baseline_defense 'Epic' --epic_subset_size 0.3;
)





### Node4: R18 HLB Baselines Narc
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --config_overrides 'R18_HLB';
python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB';

### Node1: EBM Filter
(
for i in 1000 750 500 150; do
    for j in 0.8 0.9; do
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 --ebm_filter $j \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]";
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --ebm_filter $j \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]";
    done;
done;
)

### Node3: EBM Filter DM


### Node5: EBM Filter EBM-high
(
for i in 2000 1500 1250; do
    for j in 0.8 0.9; do
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 --ebm_filter $j \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]";
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --ebm_filter $j \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]";
    done;
done;
)


### Node2:
### Node4:


############
# Run Info #
############

### EBm Filter
--ebm_filter $j

### Narcissus

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
