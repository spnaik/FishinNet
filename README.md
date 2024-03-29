# FishinNet

Transfer learning pipeline for fish identification

## Abstract
Identify individual salmon and different fish species in big fish tanks of fish farmers.

This project is in consultation with Aquabyte, a fish-farming company. 35% of a fish farmer’s revenue is spent on feeding the fish itself. Therefore, optimizing feed efficiency can bring massive gains to a $100+Billion industry that makes up 50% of our food supply. In order to optimize feed efficiency, farmers require feeding data, biomass data and welfare data. In this project, I develop an object detection and classification pipeline to automatically track individual salmon as well as fish species. These algorithms can be leveraged by fish farmers to generate data and analytics to improve feed efficiency.

[FishiNetslides](https://docs.google.com/presentation/d/1ETEt-UlT2rp-Lxm3vSrKF9MvzMtqgklMwG4rGO7PYy8/edit#slide=id.g3cc7606ef3_0_2)

## Installation
The python package dependencies for this project can be found in installation.txt. For Installation -
*pip install -r requirement.txt*

## Three models
I have tried three models for individual fish identification and also species classification. All the codes can be found in the Notebook folder.

#### 1. Baseline model using Aquabyte data -
Since the dataset is small, I used K-Nearest neighbors model trained on the color histogram of the images with both Euclidean distance and cosine similarity as distance metric. There is an assumption here that similar images will have similar color histograms. I got an accuracy score of 0.82. This is high as the images are taken from video sequences and hence the images appear similar to each other.

#### 2. Transfer learning on Aquabyte data + youtube data -
Color Histograms are sensitive to noise and are not very robust as they don't take into account the spatial information, texture of the images. Hence, I expect CNN to perform better than baseline model. However, the problem is the limited dataset. So I created my own dataset by taking images converting youtube videos to images using opencv.

VGG16 was chosen because has accurate smaller model architechture and honestly, I didn't feel the necessity to use deeper and wider models as the number of images are less.

After defining the fully connected layer, I load the ImageNet pre-trained weight to the model
For fine-tuning purpose, I truncated the original softmax layer and replace it with my own code snippet.
I freeze the weight for the first few layers so that they remain intact throughout the fine-tuning process.
I fine-tune the model by minimizing a weighted cross entropy loss function using stochastic gradient descent algorithm.
#### 3. Public dataset
The motivation of using a third model is to design a pipeline to detect the fish in the image and classify. For this I am using a public dataset provided by Kaggle.

## Sampleimages
The sample images folder should have the following structure -

- train

    -Fish1<br />
    -Fish2<br />
    ...

- validation

    -Fish1<br />
    -Fish2<br />
     ...
## Data 
The data folder contains the csv files for both training and validation.  Also, the generated tf records created using the code generate_tfrecord.py. Due to space constraints I could not add the tf record files. To generate the files in data folder, execute the following two commands 

-python xml_to_csv.py<br />
-python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record<br />
-python generate_tfrecord.py --csv_input=data/validation_labels.csv  --output_path=data/validation.record<br />

## Example run
The custom fish detector is using tensorflow pretrained models.

python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config

## Folder explanation 
1. Notebook - all ipynb files containing the above explained models
2. Scripts  - all extra python scripts that I have used
3. data     - contains the generated csv files and tf record files




