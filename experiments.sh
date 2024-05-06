###############
# Nodes Lists #
###############

### Node1: PoisonedGen Train EBM R18 HLB
(
for i in 150; do     
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cifar10_NS[num=5000_size=32_eps=8]_nf[128]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';    
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cifar10_NS[num=5000_size=32_eps=16]_nf[128]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cifar10_nf[128]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
done;

for i in 750 1000 2000; do     
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cifar10_NS[num=5000_size=32_eps=8]_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';     
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cifar10_NS[num=5000_size=32_eps=16]_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cifar10_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
done; 
)

### Node4: PoisonedGen Train DM R18 HLB
# (
# python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[250]_nf[L]_NS[num=5000_size=32_eps=16]' --num_proc 8 --diff_T 150,125,100,75;
# python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[250]_nf[L]_NS[num=5000_size=32_eps=16]' --num_proc 8 --diff_T 150,125,100,75 --poison_type 'Narcissus';
# python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[250]_nf[L]_NS[num=5000_size=32_eps=16]' --num_proc 8 --diff_T 150,125,100,75 --poison_type 'Narcissus' --noise_eps_narcissus 16;

# # Train classifer
# for i in 150 125 100 75; do
#     python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[cifar10_DDPM[250]_nf[L]_NS[num=5000_size=32_eps=16]]_T[$i]" --poison_type 'Narcissus';
#     python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[cifar10_DDPM[250]_nf[L]_NS[num=5000_size=32_eps=16]]_T[$i]" --poison_type 'Narcissus' --noise_eps_narcissus 16;
# done
# )

### Node 3 NTG Res18 HLB Trainig
(
    for i in 100 125 150; do # 100 125 150
        python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[cifar10_DDPM[250]_nf[L]]_T[$i]" --poison_type 'NeuralTangent' --config_overrides 'R18_HLB';
        python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$i]" --poison_type 'NeuralTangent' --config_overrides 'R18_HLB';
        python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[office_home_DDPM[250]_nf[L]]_T[$i]" --poison_type 'NeuralTangent' --config_overrides 'R18_HLB';
        python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[oxford_iiit_pet_DDPM[250]_nf[L]]_T[$i]" --poison_type 'NeuralTangent' --config_overrides 'R18_HLB';
        python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[fgvc_aircraft_DDPM[250]_nf[L]]_T[$i]" --poison_type 'NeuralTangent' --config_overrides 'R18_HLB';
        python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[flowers102_DDPM[250]_nf[L]]_T[$i]" --poison_type 'NeuralTangent' --config_overrides 'R18_HLB';
        python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[food101_DDPM[250]_nf[L]]_T[$i]" --poison_type 'NeuralTangent' --config_overrides 'R18_HLB';
        python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[lfw_people_DDPM[250]_nf[L]]_T[$i]" --poison_type 'NeuralTangent' --config_overrides 'R18_HLB';
        python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[textures_DDPM[250]_nf[L]]_T[$i]" --poison_type 'NeuralTangent' --config_overrides 'R18_HLB';
    done

    for i in 700 800 850 ; do # 700 750 800 850 
        python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cifar10_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'NeuralTangent' --config_overrides 'R18_HLB';
        python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cinic10_imagenet_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'NeuralTangent' --config_overrides 'R18_HLB';
        python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[office_home_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'NeuralTangent' --config_overrides 'R18_HLB';
        python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[oxford_iiit_pet_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'NeuralTangent' --config_overrides 'R18_HLB';
        python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[fgvc_aircraft_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'NeuralTangent' --config_overrides 'R18_HLB';
        python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[flowers102_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'NeuralTangent' --config_overrides 'R18_HLB';
        python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[food101_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'NeuralTangent' --config_overrides 'R18_HLB';
        python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[lfw_people_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'NeuralTangent' --config_overrides 'R18_HLB';
        python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[textures_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'NeuralTangent' --config_overrides 'R18_HLB';
    done
)


### Node2: Purify EBM POOD
(
python3 purify.py --remote_user 'sunaybhat' --diff_model None --num_proc 8   \
    --ebm_name 'cifar10_nf[128]','cifar10_nf[128]','cifar10_nf[128]','cifar10_nf[128]','cinic10_imagenet_nf[128]','cinic10_imagenet_nf[128]','cinic10_imagenet_nf[128]','cinic10_imagenet_nf[128]' \
    --ebm_lang_steps 2000,1000,750,150,2000,1000,750,150;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --num_proc 8  \
    --ebm_name 'flowers102_nf[128]','flowers102_nf[128]','flowers102_nf[128]','flowers102_nf[128]','fgvc_aircraft_nf[128]','fgvc_aircraft_nf[128]','fgvc_aircraft_nf[128]','fgvc_aircraft_nf[128]' \
    --ebm_lang_steps 2000,1000,750,150,2000,1000,750,150;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --num_proc 8  \
    --ebm_name 'food101_nf[128]','food101_nf[128]','food101_nf[128]','food101_nf[128]','lfw_people_nf[128]','lfw_people_nf[128]','lfw_people_nf[128]','lfw_people_nf[128]' \
    --ebm_lang_steps 2000,1000,750,150,2000,1000,750,150;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --num_proc 8  \
    --ebm_name 'office_home_nf[128]','office_home_nf[128]','office_home_nf[128]','office_home_nf[128]','oxford_iiit_pet_nf[128]','oxford_iiit_pet_nf[128]','oxford_iiit_pet_nf[128]','oxford_iiit_pet_nf[128]' \
    --ebm_lang_steps 2000,1000,750,150,2000,1000,750,150;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --num_proc 8  \
    --ebm_name 'textures_nf[128]' \
    --ebm_lang_steps 2000,1000,750,150;
)



  
### Node4: Purify POOD DM (cifar10,cinic10_imagenet)
(
#train classifier
for i in 150 125 100 75; do
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[cifar10_DDPM[250]_nf[L]]_T[$i]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[cifar10_DDPM[250]_nf[L]]_T[$i]" --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$i]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[cinic10_imagenet_DDPM[250]_nf[L]]_T[$i]" --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
done
)

### Node5: Purify POOD DM (oxford, food, aircraft)
(
# Train classifer R18 HLB
for i in 150 125 100 75; do
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[oxford_iiit_pet_DDPM[250]_nf[L]]_T[$i]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[oxford_iiit_pet_DDPM[250]_nf[L]]_T[$i]" --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[food101_DDPM[250]_nf[L]]_T[$i]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[food101_DDPM[250]_nf[L]]_T[$i]" --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[fgvc_aircraft_DDPM[250]_nf[L]]_T[$i]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[fgvc_aircraft_DDPM[250]_nf[L]]_T[$i]" --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
done
)

### Node7: Purify POOD DM (flowers, office-home)
(
#train classifier
for i in 150 125 100 75; do
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[flowers102_DDPM[250]_nf[L]]_T[$i]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[flowers102_DDPM[250]_nf[L]]_T[$i]" --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[office_home_DDPM[250]_nf[L]]_T[$i]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[office_home_DDPM[250]_nf[L]]_T[$i]" --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
done
)

### Node8: Purify POOD DM (lfw_people, textures)
(
#train classifier
for i in 150 125 100 75; do
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[lfw_people_DDPM[250]_nf[L]]_T[$i]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[lfw_people_DDPM[250]_nf[L]]_T[$i]" --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[textures_DDPM[250]_nf[L]]_T[$i]" --poison_type 'Narcissus' --config_overrides 'R18_HLB';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[textures_DDPM[250]_nf[L]]_T[$i]" --poison_type 'Narcissus' --noise_eps_narcissus 16 --config_overrides 'R18_HLB';
done
)




### Node 1 Purify Poisoned EBM
(
# EBM Steps 500,300,250,150 as needed
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=8]_nf[32]' --num_proc 8 --ebm_lang_steps 500,300,250;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=8]_nf[32]' --num_proc 8 --ebm_lang_steps 500,300,250 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=16]_nf[32]' --num_proc 8 --ebm_lang_steps 500,300,250;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=16]_nf[32]' --ebm_lang_steps 500,300,250 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_nf[64]' --ebm_nf 64 --num_proc 8 --ebm_lang_steps  2000,1000,750,500,300,250,150;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_nf[64]' --ebm_nf 64 --ebm_lang_steps 500,300,250,150 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_nf[64]' --ebm_nf 64 --num_proc 8 --ebm_lang_steps 2000,1000,750 --poison_type 'Narcissus' --noise_eps_narcissus 16;


python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=8]_nf[128]' --num_proc 8 --ebm_lang_steps 500,300,250 ;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=8]_nf[128]' --num_proc 8 --ebm_lang_steps 500,300,250 --poison_type 'Narcissus' ;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=16]_nf[128]' --num_proc 8 --ebm_lang_steps 500,300,250 ;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=16]_nf[128]' --num_proc 8 --ebm_lang_steps 500,300,250 --poison_type 'Narcissus' ;

### Train Classifier
for i in 500 300 250; do
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[cifar10_NS[num=5000_size=32_eps=8]_nf[32]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[cifar10_NS[num=5000_size=32_eps=16]_nf[32]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[cifar10_nf[64]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cifar10_NS[num=5000_size=32_eps=8]_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cifar10_NS[num=5000_size=32_eps=16]_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus';
done
python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[cifar10_nf[64]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[cifar10_nf[64]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus';

for i in 2000 1000 750; do
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[cifar10_nf[64]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16;
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
