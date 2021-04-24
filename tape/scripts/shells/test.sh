#!/bin/bash
set -x

echo "Task" $1
echo "BatchSize" $2
echo "Epochs" $3
echo "Lr" $4

if [ ! $5 ];
then
    SAVE_FREQ=5
else
    SAVE_FREQ=$5
fi

# echo $SAVE_FREQ

case $1 in 
	ss )
		echo "ss..."
		python tape_eval.py transformer secondary_structure     results/secondary_structure/secondary_structure_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${3}-1))/  --metrics accuracy  --split 'casp12'    &>  ./logs/eval/eval-$1-bs-$2-ep-$3-lr-$4-casp12
		python tape_eval.py transformer secondary_structure     results/secondary_structure/secondary_structure_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${3}-1))/  --metrics accuracy  --split 'ts115'     &>  ./logs/eval/eval-$1-bs-$2-ep-$3-lr-$4-ts115
		python tape_eval.py transformer secondary_structure     results/secondary_structure/secondary_structure_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${3}-1))/  --metrics accuracy  --split 'cb513'     &>  ./logs/eval/eval-$1-bs-$2-ep-$3-lr-$4-cb513
		;;
	cc )
 		echo "cc..."
        python tape_train.py transformer contact_prediction     --from_pretrained models/out/   --batch_size $2     --num_train_epochs $3      --learning_rate $4   --output_dir ./results/contact_prediction   --warmup_steps 10     --time_or_name task-$1-bs-$2-ep-$3-lr-$4    >& ./logs/train/train-$1-bs-$2-ep-$3-lr-$4     --save_freq $SAVE_FREQ
		;;
	rh )
		echo "rh..."
		python tape_eval.py transformer remote_homology         results/remote_homology/remote_homology_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${3}-1))/		--metrics accuracy  --split test_fold_holdout           &>  ./logs/eval/eval-$1-bs-$2-ep-$3-lr-$4-fold
		python tape_eval.py transformer remote_homology         results/remote_homology/remote_homology_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${3}-1))/      --metrics accuracy  --split test_family_holdout         &>  ./logs/eval/eval-$1-bs-$2-ep-$3-lr-$4-fami
		python tape_eval.py transformer remote_homology         results/remote_homology/remote_homology_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${3}-1))/      --metrics accuracy  --split test_superfamily_holdout    &>  ./logs/eval/eval-$1-bs-$2-ep-$3-lr-$4-supr
		;;
	fl )
		echo "fl..."
		python tape_eval.py transformer fluorescence	results/fluorescence/fluorescence_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${3}-1))/		--metrics spearmanr     &>  ./logs/eval/eval-$1-bs-$2-ep-$3-lr-$4
		;;
	st )
		echo "st..."
		python tape_eval.py transformer stability		results/stability/stability_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${3}-1))/				--metrics spearmanr     &>  ./logs/eval/eval-$1-bs-$2-ep-$3-lr-$4
		;;
	* )
		echo "Usage: $name [ss|cc|rh|fl|st]"
		exit 0
		;;
esac




