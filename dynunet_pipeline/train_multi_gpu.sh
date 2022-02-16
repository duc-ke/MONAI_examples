# train step 1, with large learning rate
# although max_epochs here is 3000, my results shown that for all 5 folds,
# the best epochs is less than 400, thus maybe you can manually stop early.

lr=1e-1
fold=0
export CUDA_VISIBLE_DEVICES=3,4
num_gpus=2

python -m torch.distributed.launch --nproc_per_node=$num_gpus --nnodes=1 --node_rank=0 \
    --master_addr="localhost" --master_port=1257 \
	train.py -fold $fold -train_num_workers 4 -interval 10 -num_samples 2 \
	-learning_rate $lr -max_epochs 3000 -task_id 01 -pos_sample_num 1 \
	-expr_name baseline -tta_val True -multi_gpu True
