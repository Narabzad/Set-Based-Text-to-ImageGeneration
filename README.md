This repository contains the code for running offline evaluation of Set-Based Text-to-Image Generation.

## Evaluation on a set of generated images
To run the set of proposed evaluation metrics on a set of generated images, first clone this repository and then run  ```eval.py``` as follows:

```
python eval.py \ 
  -image_dir </path/to/folder/including/generated_images< 
  -target_image </path/to/gold/standard/target/image<
  -metric <choice of ['rbp','err']>
  -trajectory <choice of ['saliency','order']>
  -gamma <user persistency parameter default=0.8>
  -n_samples <number of sampled trajectories, default=50>
  -variety <if vairety needs to be considered when measuring relevance scores, choice of [True, False]>
```
### Example 1
```
python eval.py  \
  -image_dir generated_images \
  -target_image pandatarget.png  \
  -metric rbp   \
  -trajectory saliency   \
  -gamma 0.8  \
  -n_samples 50 \
  -variety False
```
This script will generate the following grid from [the generated images](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/tree/main/generated_images) give you the following outputs:

![alt text](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/blob/main/grids/grid_generated_images.png)

Given [this target image](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/blob/main/pandatarget.png),the script will evaluate RBP based on saliency trajectories as explained in the paper and show the following outputs:

```
grid of images generated and saved as grids/grid_generated_images.png
1/1 [==============================] - 2s 2s/step
1/1 [==============================] - 0s 297ms/step
1/1 [==============================] - 0s 291ms/step
1/1 [==============================] - 0s 308ms/step
1/1 [==============================] - 0s 307ms/step
1/1 [==============================] - 0s 324ms/step
1/1 [==============================] - 0s 300ms/step
1/1 [==============================] - 0s 342ms/step
1/1 [==============================] - 0s 312ms/step
1/1 [==============================] - 0s 307ms/step
1/1 [==============================] - 0s 282ms/step
1/1 [==============================] - 0s 283ms/step
1/1 [==============================] - 0s 307ms/step
1/1 [==============================] - 0s 298ms/step
1/1 [==============================] - 0s 316ms/step
1/1 [==============================] - 0s 341ms/step
1/1 [==============================] - 0s 304ms/step
1/1 [==============================] - 0s 335ms/step
saliency [0.00225529 0.00182395 0.2671824  0.2021625  0.28705123 0.23540027 0.00211734 0.00200697]
The quality of the gird of generated images in generated_images directory is evaluated as :
metric rbp
variety True
trajectory saliency
evalaution: 0.636149760434303
```

### Saliency Prediction
We use the [trained visual saliency model on the web pages](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/tree/main/webpage_stonybrook_baseline) in order to predict the saliency of an image or a grid of images.
[```saliency.py```](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/blob/main/saliency.py) provide neccessary functions to preprocess an image and predict the visual saliency.

For example, the following command, will predict the saliency of a single image:

```  
python saliency.py -image_dir generated_images/i1.png
```

### Relevance
[```inception.py```](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/blob/main/inception.py) provide neccessary function to embed the images using InceptionV3 model and find the relevance score w.r.t a given target image. 

### Metrics
[```metrics.py```](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/blob/main/metrics.py) provide necessary functions to measure ERR, RBP and their different variations on a given list of relevance scores from a rankedlist/grid. 

