#!/bin/bash
set -x

echo "Task" $1
echo "BatchSize" $2
echo "Epochs" $3
echo "Lr" $4
echo "WS" $5

WS=$5
if [ ! $6 ];
then
    SAVE_FREQ=5
else
    SAVE_FREQ=$6
fi


for type_ in train eval
do
	for task in ss cc rh fl st
	do
		mkdir -p ./logs/$type_/$task
	done
done


case $1 in 
	ss )
		echo "ss..."
        python tape_train.py transformer secondary_structure    --from_pretrained models/out/   --batch_size $2     --num_train_epochs $3      --learning_rate $4   --output_dir ./results/secondary_structure  --warmup_steps $WS     --time_or_name task-$1-bs-$2-ep-$3-lr-$4	     --save_freq $SAVE_FREQ		>&	./logs/train/train-$1-bs-$2-ep-$3-lr-$4
		if [ -d results/secondary_structure/secondary_structure_transformer_task-$1-bs-$2-ep-$3-lr-$4/ ];then
			mkdir -p ./logs/eval/$1/eval-$1-bs-$2-ep-$3-lr-$4
		fi
		for((ep=$SAVE_FREQ;ep<$3;ep+=$SAVE_FREQ));
		do
			if [ -d results/secondary_structure/secondary_structure_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${ep}-1))/ ];then
				python tape_eval.py transformer secondary_structure     results/secondary_structure/secondary_structure_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${ep}-1))/  --metrics accuracy  --split 'casp12'    &>  ./logs/eval/$1/eval-$1-bs-$2-ep-$3-lr-$4/eval-$1-$ep-casp12
				python tape_eval.py transformer secondary_structure     results/secondary_structure/secondary_structure_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${ep}-1))/  --metrics accuracy  --split 'ts115'     &>  ./logs/eval/$1/eval-$1-bs-$2-ep-$3-lr-$4/eval-$1-$ep-ts115
				python tape_eval.py transformer secondary_structure     results/secondary_structure/secondary_structure_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${ep}-1))/  --metrics accuracy  --split 'cb513'     &>  ./logs/eval/$1/eval-$1-bs-$2-ep-$3-lr-$4/eval-$1-$ep-cb513
			fi
		done
		;;

	cc )
        echo "cc..."
        python tape_train.py transformer contact_prediction     --from_pretrained models/out/   --batch_size $2     --num_train_epochs $3      --learning_rate $4   --output_dir ./results/contact_prediction   --warmup_steps $WS     --time_or_name task-$1-bs-$2-ep-$3-lr-$4	     --save_freq $SAVE_FREQ		>&	./logs/train/$1/train-$1-bs-$2-ep-$3-lr-$4

        if [ -d results/contact_prediction/contact_prediction_transformer_task-$1-bs-$2-ep-$3-lr-$4/ ];then
            mkdir -p ./logs/eval/$1/eval-$1-bs-$2-ep-$3-lr-$4
        fi
        for((ep=$SAVE_FREQ;ep<$3;ep+=$SAVE_FREQ));
        do
            if [ -d results/contact_prediction/contact_prediction_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${ep}-1))/ ];then
                python tape_eval.py transformer contact_prediction      results/contact_prediction/contact_prediction_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${ep}-1))/  --metrics accuracy L5    --batch_size 8      &>  ./logs/eval/$1/eval-$1-bs-$2-ep-$3-lr-$4/eval-$1-$ep
            fi
        done
		;;

	rh )
		echo "rh..."
        python tape_train.py transformer remote_homology        --from_pretrained models/out/   --batch_size $2     --num_train_epochs $3      --learning_rate $4   --output_dir ./results/remote_homology      --warmup_steps $WS     --time_or_name task-$1-bs-$2-ep-$3-lr-$4	     --save_freq $SAVE_FREQ		>&	./logs/train/train-$1-bs-$2-ep-$3-lr-$4
		if [ -d results/remote_homology/remote_homology_transformer_task-$1-bs-$2-ep-$3-lr-$4/ ];then
			mkdir -p ./logs/eval/$1/eval-$1-bs-$2-ep-$3-lr-$4
		fi
		for((ep=$SAVE_FREQ;ep<$3;ep+=$SAVE_FREQ));
		do
			if [ -d results/remote_homology/remote_homology_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${ep}-1))/ ];then
				python tape_eval.py transformer remote_homology         results/remote_homology/remote_homology_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${ep}-1))/		--metrics accuracy  --split test_fold_holdout           &>  ./logs/eval/$1/eval-$1-bs-$2-ep-$3-lr-$4/eval-$1-$ep-fold
				python tape_eval.py transformer remote_homology         results/remote_homology/remote_homology_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${ep}-1))/      --metrics accuracy  --split test_family_holdout         &>  ./logs/eval/$1/eval-$1-bs-$2-ep-$3-lr-$4/eval-$1-$ep-fami
				python tape_eval.py transformer remote_homology         results/remote_homology/remote_homology_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${ep}-1))/      --metrics accuracy  --split test_superfamily_holdout    &>  ./logs/eval/$1/eval-$1-bs-$2-ep-$3-lr-$4/eval-$1-$ep-supr
			fi
		done
		;;

	fl )
		echo "fl..."
        python tape_train.py transformer fluorescence	--from_pretrained models/out/   --batch_size $2     --num_train_epochs $3      --learning_rate $4   --output_dir ./results/fluorescence         --warmup_steps $WS     --time_or_name task-$1-bs-$2-ep-$3-lr-$4			     --save_freq $SAVE_FREQ		>&	./logs/train/train-$1-bs-$2-ep-$3-lr-$4
		if [ -d results/fluorescence/fluorescence_transformer_task-$1-bs-$2-ep-$3-lr-$4/ ];then
			mkdir -p ./logs/eval/$1/eval-$1-bs-$2-ep-$3-lr-$4
		fi
		for((ep=$SAVE_FREQ;ep<$3;ep+=$SAVE_FREQ));
		do
			if [ -d results/fluorescence/fluorescence_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${ep}-1))/ ];then
				python tape_eval.py transformer fluorescence	results/fluorescence/fluorescence_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${ep}-1))/		--metrics spearmanr     &>  ./logs/eval/$1/eval-$1-bs-$2-ep-$3-lr-$4/eval-$1-$ep
			fi
		done
		;;

	st )
		echo "st..."
        python tape_train.py transformer stability		--from_pretrained models/out/   --batch_size $2     --num_train_epochs $3      --learning_rate $4   --output_dir ./results/stability            --warmup_steps $WS     --time_or_name task-$1-bs-$2-ep-$3-lr-$4			     --save_freq $SAVE_FREQ		>&	./logs/train/train-$1-bs-$2-ep-$3-lr-$4
		if [ -d results/stability/stability_transformer_task-$1-bs-$2-ep-$3-lr-$4/ ];then
			mkdir -p ./logs/eval/$1/eval-$1-bs-$2-ep-$3-lr-$4
		fi
		for((ep=$SAVE_FREQ;ep<$3;ep+=$SAVE_FREQ));
		do
			if [ -d results/stability/stability_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${ep}-1))/ ];then
				python tape_eval.py transformer stability		results/stability/stability_transformer_task-$1-bs-$2-ep-$3-lr-$4/$((${ep}-1))/				--metrics spearmanr     &>  ./logs/eval/$1/eval-$1-bs-$2-ep-$3-lr-$4/eval-$1-$ep
			fi
		done
		;;

	* )
		echo "Usage: $name [ss|cc|rh|fl|st]"
		exit 0
		;;
esac
