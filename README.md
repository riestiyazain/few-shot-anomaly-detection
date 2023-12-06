# Improvement of Hierarchical Transformation-Discriminating Generative Model for Few Shot Anomaly Detection

## Changes done to [original code](https://github.com/shellysheynin/A-Hierarchical-Transformation-Discriminating-Generative-Model-for-Few-Shot-Anomaly-Detection/tree/master)
- Added new gaussian blur transformation that improves performance.
- Added code for defect visualization.
- Trained and tested model on [Industry Biscuit dataset](https://www.kaggle.com/dsv/4311115).
- Fixed various bugs.

## Abstract
The task of Anomaly detection entails identifying data samples that are unusual or too different from the data distribution. It relies on large datasets for training. We consider the work of Sheynin
et al. (2021) who propose an approach for anomaly and defect detection using only one or a few samples of normal data. Their results are reproduced; In addition to the AUC score, the precision, recall, and F1 scores are also calculated. The model is benchmarked on the Industry Biscuits dataset (Horak et al. 2022). Finally, a new data augmentation is introduced to improve performance on anomalous images with color and shape defects.

### Install dependencies

- Install torch-gpu then

```
python -m pip install -r requirements.txt
```

###  Train
To train the model on mvtec/paris/cifar/mnist/fashionMnist:

```
python main_train.py  --num_images <num_training_images>  --pos_class <normal_class_in_dataset> --index_download <index_of_training_image> --dataset <name_of_dataset>
```

Common training options:
```
--min_size                  image minimal size at the coarser scale (default 25)
--max_size                  image minimal size at the coarser scale (default 64)
--niter                     number of iterations to train per scale
--num_images                number of images to train on (1,5,10 in the paper)
--size_image                the original image size 
--fraction_defect           the number of patches to consider in defect detection (recommended arguments: 0.01-0.1)
--pos_class                 the normal class to train on
--dataset                   paris/cifar/mnist/fashionmnist/mvtec
--random_images_download    "True" if training random images from the normal class (otherwise, specify the index of the training image in --index_download)
--devices_ids                for 10shot we have used --device_ids = 0 1 
```

## Acknowledgements
- Original work done by Shelly Sheynin, Sagie Benaim, Lior Wolf. Repository link: [GitHub Repository](https://github.com/shellysheynin/A-Hierarchical-Transformation-Discriminating-Generative-Model-for-Few-Shot-Anomaly-Detection/tree/master), [Project](https://shellysheynin.github.io/HTDG/), [Arxiv](https://arxiv.org/abs/2104.14535) 
- The implementation is based on the architecture of [SinGAN](https://github.com/tamarott/SinGAN)




