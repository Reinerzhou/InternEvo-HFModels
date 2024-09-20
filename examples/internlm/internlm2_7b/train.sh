export PYTHONPATH=./huggingface_model:$PYTHONPATH
export PARTITION="llm_s"
export GPU_NUMS=8
# srun -p $PARTITION -N $(expr $GPU_NUMS / 8) -n $GPU_NUMS --ntasks-per-node=8 --gpus-per-task=1 python ./examples/internlm/internlm2_7b/train.py --config ./examples/internlm/internlm2_7b/config.py

torchrun --nnodes=1 --nproc_per_node=8 --master_port=22502 ./examples/internlm/internlm2_7b/train.py --config ./examples/internlm/internlm2_7b/config.py --launcher "torch"

# bash examples/internlm/internlm2_7b/train.sh
