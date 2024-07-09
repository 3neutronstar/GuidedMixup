NGPUS=$1
MASTERPORT=29501
python3 -m torch.distributed.launch --nproc_per_node $NGPUS --master_port $MASTERPORT ../main.py train --ddp "${@:2}"
