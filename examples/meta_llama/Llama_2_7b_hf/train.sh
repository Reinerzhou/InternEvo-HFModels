export PYTHONPATH=./huggingface_model:$PYTHONPATH
export PARTITION="llm_s"
export GPU_NUMS=8
# srun -p $PARTITION -N $(expr $GPU_NUMS / 8) -n $GPU_NUMS --ntasks-per-node=8 --gpus-per-task=1 python ./examples/meta_llama/Llama_2_7b_hf/train.py --config ./examples/meta_llama/Llama_2_7b_hf/config.py

torchrun --nnodes=1 --nproc_per_node=8 --master_port=22503 ./examples/meta_llama/Llama_2_7b_hf/train.py --config ./examples/meta_llama/Llama_2_7b_hf/config.py --launcher "torch"

bash examples/meta_llama/Llama_2_7b_hf/train.sh
