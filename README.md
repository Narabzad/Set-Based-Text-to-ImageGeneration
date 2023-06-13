This repo contains the code for running offline evaluation of Set-Based Text-to-Image Generation.

## Evaluation on set of generated images
To run the set of propsoed evaluation metrics on a set of generated images, first clone this repository and then run ```eval.py``` as follows:

```
python eval.py \ 
  -image_dir /path/to/folder/including/generated_images 
  -target_image /path/to/gold/standard/target/image
  -metric <choice of ['rbp','err']>
  -trajectory <choice of ['saliency','order']>
  -gamma <user persistensy parameter default=0.8>
  -n_samples <number of sampled trajectories, default=50>
```
For example:

```
python eval.py  \
  -image_dir generated_images \
  -target_image pandatarget.png  \
  -metric rbp   \
  -trajectory saliency   \
  -gamma 0.8  \
  -n_samples 50
```
This script will generated the following grid from [the generated images](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/tree/main/generated_images) give you the following outputs:

![alt text](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/blob/main/grids/grid_generated_images.png)

Given [thistarget image](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/blob/main/pandatarget.png), the script will evaluate rbp based on saliency trajectories as explained in the paper and will show the following outputs: 

```
grid of images generated and saved as grids/grid_generated_images.png
saliency [0.00225529 0.00182395 0.2671824  0.2021625  0.28705123 0.23540027
 0.00211734 0.00200697]
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
relevance [0.7808417081832886, 0.7563698887825012, 0.7388613820075989, 0.7263504862785339, 0.749413013458252, 0.7648179531097412, 0.7904155850410461, 0.7195844054222107]
saliency rbp
<class 'str'>
The quality of the gird of generated images in generated_images is evaluated as :0.7573248372242919
```

### Saliency Prediction
We use the [trained visual saliency model on the web pages](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/tree/main/webpage_stonybrook_baseline) in order to predict the saliency of an image or a grid of images.
[```saliency.py```](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/blob/main/saliency.py) provide neccessary functions to preprocess an image and predict the visual saliency.

### Relevance
[```inception.py```](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/blob/main/inception.py) provide neccessary function to embed the images using InceptionV3 model and find the relevance score w.r.t a given target image. 

### Metrics
[```metrics.py```](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/blob/main/metrics.py) provide necessary functions to measure ERR, RBP and their different variations on a given list of relevance scores from a rankedlist/grid. 

