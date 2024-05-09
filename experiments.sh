###############
# Nodes Lists #
###############

python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --data_key 'EBMSNGAN32[cifar10_nf[128]]_Steps[150]_T[0.0001]'  --ebm_filter 0.1;

### Node1: EBM Filter
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_filter 0.1 --v; 

### Node2: Reps
(
python3 purify.py --remote_user 'sunaybhat' --purify_reps 2 --num_proc 8 \
    --ebm_lang_steps 50,50,50,50,25,25,25,25 \
    --diff_T 75,50,25,10,75,50,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 2 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 \
    --ebm_lang_steps 50,50,50,50,25,25,25,25 \
    --diff_T 75,50,25,10,75,50,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 3 --num_proc 8 \
    --ebm_lang_steps 50,50,50,50,25,25,25,25 \
    --diff_T 75,50,25,10,75,50,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 3 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 \
    --ebm_lang_steps 50,50,50,50,25,25,25,25 \
    --diff_T 75,50,25,10,75,50,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 4 --num_proc 8 \
    --ebm_lang_steps 50,50,50,50,25,25,25,25 \
    --diff_T 75,50,25,10,75,50,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 4 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 \
    --ebm_lang_steps 50,50,50,50,25,25,25,25 \
    --diff_T 75,50,25,10,75,50,25,10;

# Train Classifier
for i in 75 50 25 10; do
    for j in 2 3 4; do
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[25]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$i]_reps$j";
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[50]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$i]_reps$j";
    done;
done;
)

### Node 4: Mix Purify NTGA
(
python3 purify.py --remote_user 'sunaybhat' --purify_reps 2 --num_proc 8 \
    --ebm_lang_steps 250,250,250,250,150,150,150,150 \
    --diff_T 75,50,25,10,75,50,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 2 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 \
    --ebm_lang_steps 250,250,250,250,150,150,150,150 \
    --diff_T 75,50,25,10,75,50,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 3 --num_proc 8 \
    --ebm_lang_steps 250,250,250,250,150,150,150,150 \
    --diff_T 75,50,25,10,75,50,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 3 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 \
    --ebm_lang_steps 250,250,250,250,150,150,150,150 \
    --diff_T 75,50,25,10,75,50,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 4 --num_proc 8 \
    --ebm_lang_steps 250,250,250,250,150,150,150,150 \
    --diff_T 75,50,25,10,75,50,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 4 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 \
    --ebm_lang_steps 250,250,250,250,150,150,150,150 \
    --diff_T 75,50,25,10,75,50,25,10;
)


### Node7: EBM+Diff Purify Narc Eps 16
(
python3 purify.py --remote_user 'sunaybhat' --purify_reps 2 --num_proc 8 \
    --ebm_lang_steps 500,500,500,500,350,350,350,350 \
    --diff_T 75,50,25,10,75,50,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 2 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 \
    --ebm_lang_steps 500,500,500,500,350,350,350,350 \
    --diff_T 75,50,25,10,75,50,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 3 --num_proc 8 \
    --ebm_lang_steps 500,500,500,500,350,350,350,350 \
    --diff_T 75,50,25,10,75,50,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 3 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 \
    --ebm_lang_steps 500,500,500,500,350,350,350,350 \
    --diff_T 75,50,25,10,75,50,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 4 --num_proc 8 \
    --ebm_lang_steps 500,500,500,500,350,350,350,350 \
    --diff_T 75,50,25,10,75,50,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 4 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 \
    --ebm_lang_steps 500,500,500,500,350,350,350,350 \
    --diff_T 75,50,25,10,75,50,25,10;
)

### Node8: EBM+Diff Purify Narc Eps 16
(
python3 purify.py --remote_user 'sunaybhat' --purify_reps 2 --num_proc 8 \
    --ebm_lang_steps 10,10,25,25,1000,1000,750,750 \
    --diff_T 125,100,125,100,25,10,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 2 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 \
    --ebm_lang_steps 10,10,25,25,1000,1000,750,750 \
    --diff_T 125,100,125,100,25,10,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 3 --num_proc 8 \
    --ebm_lang_steps 10,10,25,25,1000,1000,750,750 \
    --diff_T 125,100,125,100,25,10,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 3 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 \
    --ebm_lang_steps 10,10,25,25,1000,1000,750,750 \
    --diff_T 125,100,125,100,25,10,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 4 --num_proc 8 \
    --ebm_lang_steps 10,10,25,25,1000,1000,750,750 \
    --diff_T 125,100,125,100,25,10,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 4 --poison_type 'Narcissus' --noise_eps_narcissus 16 --num_proc 8 \
    --ebm_lang_steps 10,10,25,25,1000,1000,750,750 \
    --diff_T 125,100,125,100,25,10,25,10;
)


### Node3: NTGA Mix Extremes
(
python3 purify.py --remote_user 'sunaybhat' --purify_reps 2 --poison_type 'NeuralTangent' --num_proc 8 \
    --ebm_lang_steps 10,10,25,25,1000,1000,750,750 \
    --diff_T 125,100,125,100,25,10,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 3 --poison_type 'NeuralTangent' --num_proc 8 \
    --ebm_lang_steps 10,10,25,25,1000,1000,750,750 \
    --diff_T 125,100,125,100,25,10,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 4 --poison_type 'NeuralTangent' --num_proc 8 \
    --ebm_lang_steps 10,10,25,25,1000,1000,750,750 \
    --diff_T 125,100,125,100,25,10,25,10;
)

### Node5: Mix Purify NTGA
(
python3 purify.py --remote_user 'sunaybhat' --purify_reps 2 --poison_type 'NeuralTangent' --num_proc 8 \
    --ebm_lang_steps 250,250,250,250,150,150,150,150 \
    --diff_T 75,50,25,10,75,50,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 3 --poison_type 'NeuralTangent' --num_proc 8 \
    --ebm_lang_steps 250,250,250,250,150,150,150,150 \
    --diff_T 75,50,25,10,75,50,25,10;
python3 purify.py --remote_user 'sunaybhat' --purify_reps 4 --poison_type 'NeuralTangent' --num_proc 8 \
    --ebm_lang_steps 250,250,250,250,150,150,150,150 \
    --diff_T 75,50,25,10,75,50,25,10;
)


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
