#!/bin/bash

#num_folds_1=4
data_name="Dpendulum_I"
i=0

#for i in $(seq 0 1 $num_folds_1)
#do
    echo $i
    for k in $(seq 0 1 2)
    do
        data_path=$data_name"/fold_"$i"/$data_name""_fold$i""_side""$k""_data"
        echo $data_path
        model_name=$data_name"_fold$i""_side""$k"
        echo $model_name
        poetry run python run_gruode.py --model_name=$model_name --dataset=$data_path --hidden_size=50
    done
#done

