export PYTHONPATH=./huggingface_model:$PYTHONPATH
export PARTITION="llm_s"
export GPU_NUMS=8
# srun -p $PARTITION -N $(expr $GPU_NUMS / 8) -n $GPU_NUMS --ntasks-per-node=8 --gpus-per-task=1 python ./examples/Qwen/Qwen2_7B/train.py --config ./examples/Qwen/Qwen2_7B/config.py
torchrun --nnodes=1 --nproc_per_node=8 --master_port=22502 ./examples/Qwen/Qwen2_7B/train.py --config ./examples/Qwen/Qwen2_7B/config.py --launcher "torch"

bash examples/Qwen/Qwen2_7B/train.sh
