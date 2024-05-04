###############
# Nodes Lists #
###############

### Node 7 Purify GM
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cinic10_imagenet_nf[32]' --ebm_lang_steps 150 --poison_type 'GradientMatching';

python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_model None --poison_type 'GradientMatching';
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_model None;

# Train Classifier
python3 train_classifier.py --remote_user 'sunaybhat' --data_key "Baseline" --poison_type 'GradientMatching';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[cinic10_imagenet_nf[32]]_Steps[150]_T[0.0001]" --poison_type 'GradientMatching';


### Node 1Purify Poisoned EBM
(
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=8]_nf[32]' --num_proc 8 --ebm_lang_steps 2000,1000,750,150;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=8]_nf[32]' --ebm_lang_steps 150 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=8]_nf[32]' --num_proc 8 --ebm_lang_steps 2000,1000,750 --poison_type 'Narcissus' --noise_eps_narcissus 16;

python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=16]_nf[32]' --num_proc 8 --ebm_lang_steps 2000,1000,750,150;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=16]_nf[32]' --ebm_lang_steps 150 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=16]_nf[32]' --num_proc 8 --ebm_lang_steps 2000,1000,750 --poison_type 'Narcissus' --noise_eps_narcissus 16;

python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=8]_nf[128]' --num_proc 8 --ebm_lang_steps 2000,1000,750,150 --ebm_model 'EBMSNGAN32' --ebm_nf 128;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=8]_nf[128]' --ebm_lang_steps 150 --poison_type 'Narcissus' --ebm_model 'EBMSNGAN32' --ebm_nf 128;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=8]_nf[128]' --num_proc 8 --ebm_lang_steps 2000,1000,750 --poison_type 'Narcissus' --noise_eps_narcissus 16 --ebm_model 'EBMSNGAN32' --ebm_nf 128;

python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=16]_nf[128]' --num_proc 8 --ebm_lang_steps 2000,1000,750,150 --ebm_model 'EBMSNGAN32' --ebm_nf 128;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=16]_nf[128]' --ebm_lang_steps 150 --poison_type 'Narcissus' --ebm_model 'EBMSNGAN32' --ebm_nf 128;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --ebm_name 'cifar10_NS[num=5000_size=32_eps=16]_nf[128]' --num_proc 8 --ebm_lang_steps 2000,1000,750 --poison_type 'Narcissus' --noise_eps_narcissus 16 --ebm_model 'EBMSNGAN32' --ebm_nf 128;

# Train classifer
# python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[cifar10_NS[num=5000_size=32_eps=8]_nf[32]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[cifar10_NS[num=5000_size=32_eps=16]_nf[32]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cifar10_NS[num=5000_size=32_eps=8]_nf[128]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus';
python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cifar10_NS[num=5000_size=32_eps=16]_nf[128]]_Steps[150]_T[0.0001]" --poison_type 'Narcissus';

for i in 2000 1000 750; do
    # python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[cifar10_NS[num=5000_size=32_eps=8]_nf[32]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16;
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBM[cifar10_NS[num=5000_size=32_eps=16]_nf[32]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16;
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cifar10_NS[num=5000_size=32_eps=8]_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16;
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "EBMSNGAN32[cifar10_NS[num=5000_size=32_eps=16]_nf[128]]_Steps[$i]_T[0.0001]" --poison_type 'Narcissus' --noise_eps_narcissus 16;
done
)

### Node2: Purify EBM POOD
(
python3 purify.py --remote_user 'sunaybhat' --diff_model None --num_proc 8 --ebm_nf 64 \
    --ebm_name 'cifar10_nf[64]','cifar10_nf[64]','cifar10_nf[64]','cifar10_nf[64]','fgvc_aircraft_nf[64]','fgvc_aircraft_nf[64]','fgvc_aircraft_nf[64]','fgvc_aircraft_nf[64]' \
    --ebm_lang_steps 2000,1000,750,150,2000,1000,750,150;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --num_proc 8 \
    --ebm_name 'flowers102_nf[32]','flowers102_nf[32]','flowers102_nf[32]','flowers102_nf[32]','cinic10_imagenet_nf[32]','cinic10_imagenet_nf[32]','cinic10_imagenet_nf[32]','cinic10_imagenet_nf[32]' \
    --ebm_lang_steps 2000,1000,750,150,2000,1000,750,150;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --num_proc 8 \
    --ebm_name 'food101_nf[32]','food101_nf[32]','food101_nf[32]','food101_nf[32]','lfw_people_nf[32]','lfw_people_nf[32]','lfw_people_nf[32]','lfw_people_nf[32]' \
    --ebm_lang_steps 2000,1000,750,150,2000,1000,750,150;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --num_proc 8 \
    --ebm_name 'office_home_nf[32]','office_home_nf[32]','office_home_nf[32]','office_home_nf[32]','oxford_iiit_pet_nf[32]','oxford_iiit_pet_nf[32]','oxford_iiit_pet_nf[32]','oxford_iiit_pet_nf[32]' \
    --ebm_lang_steps 2000,1000,750,150,2000,1000,750,150;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --num_proc 8 \
    --ebm_name 'textures_nf[32]' \
    --ebm_lang_steps 2000,1000,750,150;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --num_proc 8  --ebm_model 'EBMSNGAN32' --ebm_nf 128 \
    --ebm_name 'cifar10_nf[128]','cifar10_nf[128]','cifar10_nf[128]','cifar10_nf[128]','cinic10_imagenet_nf[128]','cinic10_imagenet_nf[128]','cinic10_imagenet_nf[128]','cinic10_imagenet_nf[128]' \
    --ebm_lang_steps 2000,1000,750,150,2000,1000,750,150;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --num_proc 8 --ebm_model 'EBMSNGAN32' --ebm_nf 128 \
    --ebm_name 'flowers102_nf[128]','flowers102_nf[128]','flowers102_nf[128]','flowers102_nf[128]','fgvc_aircraft_nf[128]','fgvc_aircraft_nf[128]','fgvc_aircraft_nf[128]','fgvc_aircraft_nf[128]' \
    --ebm_lang_steps 2000,1000,750,150,2000,1000,750,150;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --num_proc 8 --ebm_model 'EBMSNGAN32' --ebm_nf 128 \
    --ebm_name 'food101_nf[128]','food101_nf[128]','food101_nf[128]','food101_nf[128]','lfw_people_nf[128]','lfw_people_nf[128]','lfw_people_nf[128]','lfw_people_nf[128]' \
    --ebm_lang_steps 2000,1000,750,150,2000,1000,750,150;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --num_proc 8 --ebm_model 'EBMSNGAN32' --ebm_nf 128 \
    --ebm_name 'office_home_nf[128]','office_home_nf[128]','office_home_nf[128]','office_home_nf[128]','oxford_iiit_pet_nf[128]','oxford_iiit_pet_nf[128]','oxford_iiit_pet_nf[128]','oxford_iiit_pet_nf[128]' \
    --ebm_lang_steps 2000,1000,750,150,2000,1000,750,150;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --num_proc 8 --ebm_model 'EBMSNGAN32' --ebm_nf 128 \
    --ebm_name 'textures_nf[128]' \
    --ebm_lang_steps 2000,1000,750,150;
)



### Node3: NTG Posion Purify and Train
(
python3 purify.py --remote_user 'sunaybhat' --diff_model None --poison_type 'NeuralTangent' --num_proc 8 --ebm_nf 64 \
    --ebm_name 'cifar10_nf[64]','cifar10_nf[64]','cifar10_nf[64]','cifar10_nf[64]','fgvc_aircraft_nf[64]','fgvc_aircraft_nf[64]','fgvc_aircraft_nf[64]','fgvc_aircraft_nf[64]' \
    --ebm_lang_steps 850,800,750,700,850,800,750,700;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --poison_type 'NeuralTangent' --num_proc 8 \
    --ebm_name 'flowers102_nf[32]','flowers102_nf[32]','flowers102_nf[32]','flowers102_nf[32]','cinic10_imagenet_nf[32]','cinic10_imagenet_nf[32]','cinic10_imagenet_nf[32]','cinic10_imagenet_nf[32]' \
    --ebm_lang_steps 850,800,750,700,850,800,750,700;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --poison_type 'NeuralTangent' --num_proc 8 \
    --ebm_name 'food101_nf[32]','food101_nf[32]','food101_nf[32]','food101_nf[32]','lfw_people_nf[32]','lfw_people_nf[32]','lfw_people_nf[32]','lfw_people_nf[32]' \
    --ebm_lang_steps 850,800,750,700,850,800,750,700;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --poison_type 'NeuralTangent' --num_proc 8 \
    --ebm_name 'office_home_nf[32]','office_home_nf[32]','office_home_nf[32]','office_home_nf[32]','oxford_iiit_pet_nf[32]','oxford_iiit_pet_nf[32]','oxford_iiit_pet_nf[32]','oxford_iiit_pet_nf[32]' \
    --ebm_lang_steps 850,800,750,700,850,800,750,700;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --poison_type 'NeuralTangent' --num_proc 8 \
    --ebm_name 'textures_nf[32]' \
    --ebm_lang_steps 850,800,750,700;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --poison_type 'NeuralTangent' --num_proc 8  --ebm_model 'EBMSNGAN32' --ebm_nf 128 \
    --ebm_name 'cifar10_nf[128]','cifar10_nf[128]','cifar10_nf[128]','cifar10_nf[128]','cinic10_imagenet_nf[128]','cinic10_imagenet_nf[128]','cinic10_imagenet_nf[128]','cinic10_imagenet_nf[128]' \
    --ebm_lang_steps 850,800,750,700,850,800,750,700;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --poison_type 'NeuralTangent' --num_proc 8 --ebm_model 'EBMSNGAN32' --ebm_nf 128 \
    --ebm_name 'flowers102_nf[128]','flowers102_nf[128]','flowers102_nf[128]','flowers102_nf[128]','fgvc_aircraft_nf[128]','fgvc_aircraft_nf[128]','fgvc_aircraft_nf[128]','fgvc_aircraft_nf[128]' \
    --ebm_lang_steps 850,800,750,700,850,800,750,700;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --poison_type 'NeuralTangent' --num_proc 8 --ebm_model 'EBMSNGAN32' --ebm_nf 128 \
    --ebm_name 'food101_nf[128]','food101_nf[128]','food101_nf[128]','food101_nf[128]','lfw_people_nf[128]','lfw_people_nf[128]','lfw_people_nf[128]','lfw_people_nf[128]' \
    --ebm_lang_steps 850,800,750,700,850,800,750,700;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --poison_type 'NeuralTangent' --num_proc 8 --ebm_model 'EBMSNGAN32' --ebm_nf 128 \
    --ebm_name 'office_home_nf[128]','office_home_nf[128]','office_home_nf[128]','office_home_nf[128]','oxford_iiit_pet_nf[128]','oxford_iiit_pet_nf[128]','oxford_iiit_pet_nf[128]','oxford_iiit_pet_nf[128]' \
    --ebm_lang_steps 850,800,750,700,850,800,750,700;
python3 purify.py --remote_user 'sunaybhat' --diff_model None --poison_type 'NeuralTangent' --num_proc 8 --ebm_model 'EBMSNGAN32' --ebm_nf 128 \
    --ebm_name 'textures_nf[128]' \
    --ebm_lang_steps 850,800,750,700;
)


### Node4: Purify DM
(
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[250]_nf[L]_NS[num=5000_size=32_eps=16]' --num_proc 8 --diff_T 150,125,100,75;
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[250]_nf[L]_NS[num=5000_size=32_eps=16]' --num_proc 8 --diff_T 150,125,100,75 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'cifar10_DDPM[250]_nf[L]_NS[num=5000_size=32_eps=16]' --num_proc 8 --diff_T 150,125,100,75 --poison_type 'Narcissus' --noise_eps_narcissus 16;

# Train classifer
for i in 150 125 100 75; do
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[cifar10_DDPM[250]_nf[L]_NS[num=5000_size=32_eps=16]]_T[$i]" --poison_type 'Narcissus';
    python3 train_classifier.py --remote_user 'sunaybhat' --data_key "DM_UNET[cifar10_DDPM[250]_nf[L]_NS[num=5000_size=32_eps=16]]_T[$i]" --poison_type 'Narcissus' --noise_eps_narcissus 16;
done
)

### Node5: Purify POOD DM
(
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'oxford_iiit_pet_DDPM[250]_nf[L]' --num_proc 8 --diff_T 150,125,100,75;
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'oxford_iiit_pet_DDPM[250]_nf[L]' --num_proc 8 --diff_T 150,125,100,75 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'oxford_iiit_pet_DDPM[250]_nf[L]' --num_proc 8 --diff_T 150,125,100,75 --poison_type 'Narcissus' --noise_eps_narcissus 16;

python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'food101_DDPM[250]_nf[L]' --num_proc 8 --diff_T 150,125,100,75;
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'food101_DDPM[250]_nf[L]' --num_proc 8 --diff_T 150,125,100,75 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'food101_DDPM[250]_nf[L]' --num_proc 8 --diff_T 150,125,100,75 --poison_type 'Narcissus' --noise_eps_narcissus 16;

python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'fgvc_aircraft_DDPM[250]_nf[L]' --num_proc 8 --diff_T 150,125,100,75;
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'fgvc_aircraft_DDPM[250]_nf[L]' --num_proc 8 --diff_T 150,125,100,75 --poison_type 'Narcissus';
python3 purify.py --remote_user 'sunaybhat' --ebm_model None --diff_name 'fgvc_aircraft_DDPM[250]_nf[L]' --num_proc 8 --diff_T 150,125,100,75 --poison_type 'Narcissus' --noise_eps_narcissus 16;
)




############################
# Setup Node and Copy Data #
############################

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
