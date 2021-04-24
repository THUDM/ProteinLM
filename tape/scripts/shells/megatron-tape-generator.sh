#!/bin/bash
# set -x

HOME=$PWD
echo "Working in" $HOME

if [ ! $1 ];
then
    MEGA_CKPT=$HOME/models/mega
else
    MEGA_CKPT=$1
fi

while true; do
     read -p "Be sure to have tape config in $HOME/converter/config.json " yn
     case $yn in
            [Yy]* ) echo "Begin transferring and training..."; break;;
            [Nn]* ) exit;;
            * ) echo "Please answer yes (Y/y) or no (N/n).";;
    esac
done


# PART0 Generate Untrained Tape
echo "Generate untrained model in" $HOME"/models/untrained_tape"
cd $HOME
mkdir -p models
mkdir -p models/mega
mkdir -p models/untrained_tape
mkdir -p models/out
python tape_train.py transformer masked_language_modeling --batch_size 8 --num_train_epochs 1 --model_config_file $HOME/converter/config.json --time_or_name untrained --force_save $HOME/models/untrained_tape


# PART1 Model Transfer
echo "Transfer megatron params to untrained tape"
cd ./models
mkdir -p out

source $HOME/scripts/shells/activate_torch1.7.sh
python $HOME/converter/megatron-converter.py  -src $MEGA_CKPT/model_optim_rng.pt  -dst $HOME/models/untrained_tape/0/pytorch_model.bin  -out $HOME/models/out/pytorch_model.bin -dtp torch.float32 -hidden 1024 -heads 16 -layers 16
cp untrained_tape/0/config.json out/config.json
source $HOME/scripts/shells/activate_base.sh
