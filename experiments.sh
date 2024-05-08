###############
# Nodes Lists #
###############


### Node1
## Node4: PoisonedGen Train DM R18 HLB

### Node2: Purify EBM POOD
(
# Train Classifier

python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cifar10_nf[128]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[flowers102_nf[128]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[fgvc_aircraft_nf[128]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[food101_nf[128]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[lfw_people_nf[128]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[office_home_nf[128]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[oxford_iiit_pet_nf[128]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[textures_nf[128]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';

for i in 2000 1000 750; do
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cifar10_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[flowers102_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[fgvc_aircraft_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[food101_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[lfw_people_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[office_home_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[oxford_iiit_pet_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[textures_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
done
)


### Node 3 NTG EBM_DM Testing
(
# Train Classifier
for i in 500 300; do
    for j in 100 75 50 25; do
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'NeuralTangent' \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$j]_ReverseOnly";
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'NeuralTangent' \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$j]";
    done
done
)

### Node5: Mix Purify NTGA
(
# Train Classifier
for i in 750 650; do
    for j in 150 125 100 75; do
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'NeuralTangent' \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$j]_ReverseOnly";
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'NeuralTangent' \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$j]";
    done
done
)

### Node7: EBM+Diff Purify Narc Eps 16
(
# Train Classifier
for i in 2000 1500; do
    for j in 150 125 100 75; do
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$j]_ReverseOnly";
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$j]";
    done
done
)

### Node8: EBM+Diff Purify Narc Eps 16
(
# Train Classifier
for i in 1000 800; do
    for j in 150 125 100 75; do
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$j]_ReverseOnly";
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$j]";
    done
done
)

### Node9: EBM+Diff Purify Narc Eps 16
(
# Train Classifier
for i in 600 400; do
    for j in 150 125 100 75; do
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$j]_ReverseOnly";
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$j]";
    done
done

for i in 250 150; do
    for j in 100 75 50 25; do
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$j]_ReverseOnly";
        python3 train_classifier.py --remote_user 'sunaybhat' --poison_type 'Narcissus' --noise_eps_narcissus 16 \
            --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]_DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$j]";
    done
done
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
