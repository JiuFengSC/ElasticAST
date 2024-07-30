dataset=epic-sound
# full or balanced for audioset
set=full

bal=full
lr=1e-4
epoch=30
tr_data=./data/datafiles/EPIC_Sounds_train.json
lrscheduler_start=2
lrscheduler_step=1
lrscheduler_decay=0.75
wa_start=1
wa_end=5

te_data=./egs/epic-sound/data/datafiles/EPIC_Sounds_val.json

freqm=48
timem=192
mixup=0
# corresponding to overlap of 6 for 16*16 patches
fstride=16
tstride=16
batch_size=12

# if we don't change the windon size
# 1024 
dataset_mean=-8.820385
dataset_std=3.7790387

# 400 
# dataset_mean=-8.660809
# dataset_std=3.878928
# window_change=False


# if we change the windon size accordingly
# dataset_mean=-4.40
# dataset_std=4.41856705
# window_change=True

audio_length=1024
Fshift=5
window_size=10
noise=False

metrics=acc
loss=CE
warmup=True
wa=True

seed=1337
exp=Epic_AST_IM-bs${batch_size}-lr${lr}-${freqm}-${timem}-epic-real${audio_length}-offcial-loader_seed${seed}
# exp=Epic_AST_noImageNet_DeiT384_seed${seed}_lr${lr}
exp_dir=/mnt/bear2/users/fj/ElasticAST/ast/${dataset}/${exp}

mkdir -p $exp_dir

CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_DISABLE=1 python -W ignore ../../src/train_AST-epic-sound.py \
--dataset ${dataset} \
--data_train ${tr_data} --data_val ${te_data} --exp_dir $exp_dir \
--label_csv /home/jfeng/FJ/ElasticAST/egs/epic-sound/data/class_labels_indices.csv --n_class 44 \
--lr $lr --n_epochs ${epoch} --batch_size $batch_size \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std}  --noise ${noise} \
--audio_length ${audio_length} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--Fshift ${Fshift} --window_size ${window_size} \
--seed ${seed} \
--exp_name ${exp} --exp_dir ${exp_dir} \
--skip_norm \
--wandb \
