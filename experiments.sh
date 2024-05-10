###############
# Nodes Lists #
###############

### Node5

### Node1: EBM Filter
(
python3 purify.py --remote_user 'sunaybhat' --diff_model None --num_proc 8 \
    --ebm_lang_steps 1000,750,500,150;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --num_proc 8 --poison_type 'Narcissus' --noise_eps_narcissus 16 \
    --ebm_lang_steps 1000,750,500,150;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --num_proc 8 --poison_type 'Narcissus' \
    --ebm_lang_steps 1000,750,500,150;

# Train Narc Basdeines
python3 train_classifier.py --remote_user 'sunaybhat';
python3 train_classifier.py --remote_user 'sunaybhat' --noise_eps_narcissus 16;

for i in 1000 750 500 150; do
    for j in 0.05 0.1 0.2 0.3; do
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 --ebm_filter $j \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]";
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --ebm_filter $j \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]";
    done;
    python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
        --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]";
    python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' \
        --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]";

done;
)

# python3 purify.py --remote_user 'sunaybhat' --purify_reps 2 --num_proc 8 \
#     --ebm_lang_steps 50,50,50,50,25,25,25,25 \
#     --diff_T 75,50,25,10,75,50,25,10;
# python3 purify.py --remote_user 'sunaybhat' --purify_reps 2 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 \
#     --ebm_lang_steps 50,50,50,50,25,25,25,25 \
#     --diff_T 75,50,25,10,75,50,25,10;

### Node2: Reps
(
python3 purify.py --remote_user 'sunaybhat' --diff_T 10 --ebm_lang_steps 10 --num_proc 8 \
    --purify_reps 9,8,7,6,5,4,3,2;
python3 purify.py --remote_user 'sunaybhat' --diff_T 10 --ebm_lang_steps 10 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 \
    --purify_reps 9,8,7,6,5,4,3,2;

python3 purify.py --remote_user 'sunaybhat' --diff_T 10 --ebm_lang_steps 25 --num_proc 8 \
    --purify_reps 9,8,7,6,5,4,3,2;
python3 purify.py --remote_user 'sunaybhat' --diff_T 10 --ebm_lang_steps 25 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 \
    --purify_reps 9,8,7,6,5,4,3,2;

python3 purify.py --remote_user 'sunaybhat' --diff_T 10 --ebm_lang_steps 50 --num_proc 8 \
    --purify_reps 9,8,7,6,5,4,3,2;
python3 purify.py --remote_user 'sunaybhat' --diff_T 10 --ebm_lang_steps 50 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 \
    --purify_reps 9,8,7,6,5,4,3,2;

python3 purify.py --remote_user 'sunaybhat' --diff_T 10 --ebm_lang_steps 75 --num_proc 8 \
    --purify_reps 9,8,7,6,5,4,3,2;
python3 purify.py --remote_user 'sunaybhat' --diff_T 10 --ebm_lang_steps 75 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 \
    --purify_reps 9,8,7,6,5,4,3,2;

python3 purify.py --remote_user 'sunaybhat' --diff_T 10 --ebm_lang_steps 100 --num_proc 8 \
    --purify_reps 9,8,7,6,5,4,3,2;
python3 purify.py --remote_user 'sunaybhat' --diff_T 10 --ebm_lang_steps 100 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 \
    --purify_reps 9,8,7,6,5,4,3,2;

# Train Classifier
for i in 10 25 50 75 100; do
    for j in 2 3 4 5 6 7 8 9; do
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[10]_reps$j";
    done;
done;
)

### Node 4: Mix Purify NTGA
(
# Train Classifier
for i in 75 50 25 10; do
    for j in 2 3 4; do
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[150]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$i]_reps$j";
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[250]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$i]_reps$j";
    done;
done;
)


### Node7: EBM+Diff Purify Narc Eps 16
(
# Train Classifier
for i in 75 50 25 10; do
    for j in 2 3 4; do
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[350]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$i]_reps$j";
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[500]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$i]_reps$j";
    done;
done;
)

### Node8: EBM+Diff Purify Narc Eps 16
(
# Train Classifier
for j in 2 3 4; do
    python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
        --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[10]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[125]_reps$j";
    python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
        --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[10]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[100]_reps$j";
    python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
        --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[25]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[125]_reps$j";
    python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
        --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[25]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[100]_reps$j";
    python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
        --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[1000]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[25]_reps$j";
    python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
        --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[1000]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[10]_reps$j";
    python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
        --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[750]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[25]_reps$j";
    python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
        --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[750]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[10]_reps$j";
done;
)




### Node3: NTGA Reps
(
python3 purify.py --remote_user 'sunaybhat' --purify_reps 2 --poison_type 'NeuralTangent' --num_proc 8 \
    --ebm_lang_steps 125,100,75,50,125,100,75,50 \
    --diff_T 50,50,50,50,25,25,25,25;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 3 --poison_type 'NeuralTangent' --num_proc 8 \
    --ebm_lang_steps 125,100,75,50,125,100,75,50 \
    --diff_T 50,50,50,50,25,25,25,25;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 4 --poison_type 'NeuralTangent' --num_proc 8 \
    --ebm_lang_steps 125,100,75,50,125,100,75,50 \
    --diff_T 50,50,50,50,25,25,25,25;

# Train Classifier
for i in 50 25; do
    for j in 2 3 4; do
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'NeuralTangent' \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[125]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$i]_reps$j";
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'NeuralTangent' \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[100]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$i]_reps$j";
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'NeuralTangent' \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[75]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$i]_reps$j";
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'NeuralTangent' \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[50]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$i]_reps$j";
    done;
done;
)

:


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
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node7:/home/sunaybhat/data/PureGen_Models/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/PureGen_Models/* sunaybhat@node8:/home/sunaybhat/data/PureGen_Models/;

# Copy Poisons Up to Node
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/Poisons/* sunaybhat@node1:/home/sunaybhat/data/Poisons/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/Poisons/* sunaybhat@node2:/home/sunaybhat/data/Poisons/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/Poisons/* sunaybhat@node3:/home/sunaybhat/data/Poisons/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/Poisons/* sunaybhat@node4:/home/sunaybhat/data/Poisons/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/Poisons/* sunaybhat@node5:/home/sunaybhat/data/Poisons/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/Poisons/* sunaybhat@node7:/home/sunaybhat/data/Poisons/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/Poisons/* sunaybhat@node8:/home/sunaybhat/data/Poisons/;
rsync -av --exclude='.DS_Store' /Users/sunaybhat/Documents/GitHub/data/Poisons/* sunaybhat@node9:/home/sunaybhat/data/Poisons/;

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
