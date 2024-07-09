#torch.distributed.launch deprecated in torch 2.0
#use torchrun instead

NGPUS=$1
#MASTERPORT=29501
torchrun --standalone --nnodes=1 --nproc-per-node=$NGPUS ../main.py train --ddp "${@:2}"
