# GuidedMixup-FGVC

To download datasets:
- [CUB200](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- [Aircraft100](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
- [Cars196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- Caltech101 (Automatically download by PyTorch Framework) </br>
We split the data into train and test as 0.8:0.2 with stratified strategy.
- [Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

## Training
### FGVC
To train the `vanilla` model(s) in the paper, run this command:

```train
python train.py --dataset_path <path_to_data>
```

To train the `guidedmixup` model(s) in the paper, run this command:

```train
python train.py --dataset_path <path_to_data> --train_mode guided-sr --mix_prob 0.5 --condition greedy --dataset <dataset_name>
```

```train
python train.py --dataset_path <path_to_data> --train_mode guided-ap --condition greedy --mix_prob 0.8 --dataset <dataset_name>
```
## Evaluation

To evaluate my model, run:

```eval
python eval.py --file_name resnet18_cub200_guided-ap_greedy
```

`--file_name` is the directory name which is located in `[our_code]\training_data`.
