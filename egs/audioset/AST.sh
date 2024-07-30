dataset=audioset
# full or balanced for audioset
set=full

bal=bal
lr=1e-5
epoch=5
tr_data=./data/datafiles/unbalanced.json
lrscheduler_start=2
lrscheduler_step=1
lrscheduler_decay=0.5
wa_start=1
wa_end=5

te_data=./data/datafiles/eval.json
freqm=48
timem=192
mixup=0.5
# corresponding to overlap of 6 for 16*16 patches
fstride=16
tstride=16
batch_size=12

# if we don't change the windon size
dataset_mean=-4.4050345 
dataset_std=4.9021754
window_change=False


# if we change the windon size accordingly
# dataset_mean=-4.40
# dataset_std=4.41856705
# window_change=True

audio_length=1024
Fshift=10
noise=False

metrics=mAP
loss=BCE
warmup=True
wa=True


seed=2024
# exp=VGGSound_AST_noImNet_baseline_0.75-10-1
exp=${dataset}_AST_pretrained_${seed}
# exp=VGGSound_AST_noImNet_baseline_LinearPatch_noNorm
# exp=debug
exp_dir=/mnt/bear1/users/fj/ElasticAST/ast/${dataset}/${exp}

mkdir -p $exp_dir

CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_DISABLE=1 python -W ignore ../../src/train_AST.py \
--dataset ${dataset} \
--data_train ${tr_data} --data_val ${te_data} --exp_dir $exp_dir \
--label_csv ./data/class_labels_indices.csv --n_class 527 \
--lr $lr --n_epochs ${epoch} --batch_size $batch_size \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std}  --noise ${noise} \
--audio_length ${audio_length} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--Fshift ${Fshift} --window_change ${window_change} \
--seed ${seed} \
--exp_name ${exp} --exp_dir ${exp_dir} \
--wandb \
# --factorized_pos \
# --no_cls attn \

# --wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} \
