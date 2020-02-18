# Painting Date Estimator
Goal of the project is developing a machine learning algorithm which is able to estimate the year in which a painting has been created, based on its appearance only.


![EstimatorResults](/images/results.jpg)
 

This code has been developed as Capstone Project for the Coursera's [Advanced Data Science with IBM Specialization](https://www.coursera.org/specializations/advanced-data-science-ibm).

## Visualizing the results
Results can be visualized using this [Notebook](./notebooks/Painting%20Date%20Estimator.ipynb).   

## Installing the dependencies
From the command prompt, type:
```bash
pip install -r requirements.txt
```

## Testing the model
Pretrained model can be tested downloading the weights from this [link](http://www.matteobustreo.com/research/data/painting-date-estimator/models_pretrained.tar.xz).
The weights should be unzipped in ```./models_pretrained``` folder.

Then:

```bash
cd src
python ./test.py
```

Input image path, pretrained model path and model architecture can be modified. Please take a look at:
```bash
python ./test.py --help
```

## Training the model
### Downloading the data
Data can be downloaded from [Painter by Numbers](https://www.kaggle.com/c/painter-by-numbers) Kaggle competition, 
after registering to Kaggle and joining the competition.

* Copy ```all_data_info.csv``` in ```./data/``` folder;
* Extract the images in ```./data/train``` and ```./data/test``` folders. 

### Running the training script
Model can be trained calling the ```train.py``` script in ```./src/``` folder:
```bash
cd src
python ./train.py
```

Training parameters, input and output directories can be modified. Please take a look at:
```bash
python ./train.py --help
```

In particular, model can be trained fine tuning Resnet152 (_default_) or SqueezeNet. In case SqueezeNet is preferred, run:
```bash
python ./train.py --select_squeezenet=True
``` 

### Monitoring training curves
Network training can be monitored via Tensorboard and it is updated once per epoch: 
```bash
tensorboard --logdir=runs
```

