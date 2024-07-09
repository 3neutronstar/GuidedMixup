# cifar100
python main.py train --model preact_resnet18 --dataset cifar100 --batch_size 128 --lr 0.1 --weight_decay 0.0001 --epochs 1200 --lr_steps 400,800

# tiny-imagenet
python main.py train --model preact_resnet18 --dataset tiny-imagenet --batch_size 256 --lr 0.1 --weight_decay 0.0001 --epochs 1200 --lr_steps 600,900

# pyramidnet
python main.py train --model pyramidnet_200_240 --dataset cifar100 --batch_size 64 --lr 0.25 --weight_decay 0.0001 --nesterov False