# Importing the relevant modules
from pyspark import *
from pyspark.sql import SQLContext
from pyspark.sql.types import FloatType

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torch import nn, device, manual_seed, optim
from torchvision import transforms

from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

import argparse
import numpy as np
import os
import time

import dataset
import model
import utils


def main(args):
    model_path = args.output_dir + args.experiment_name + '/'

    ## Configuring Spark
    spark_conf = SparkConf().setAppName("Artist Detector")
    spark_context = SparkContext.getOrCreate(conf = spark_conf)
    spark_context.setLogLevel("ERROR")
    sqlContext = SQLContext(spark_context)

    ## ETL: Preparing the data
    # Read Input annotations and store in dataframe
    df = sqlContext.read.format("csv").option("header", "true").load(args.csv_filepath)
    df.createOrReplaceTempView('artists')

    # Splitting train and test sets
    df_train = sqlContext.sql("""SELECT * from artists where in_train='True'""")
    df_train.createOrReplaceTempView('train_set')

    df_test = sqlContext.sql("""SELECT * from artists where in_train='False'""")
    df_test.createOrReplaceTempView('test_set')

    # Defining date cleansing function
    sqlContext.udf.register("kdo", lambda s: utils.keep_date_only(s), FloatType())

    # Cleaning inputs and selecting relevant columns for generating Train, Validation and Test Set
    train_val_df_ori = sqlContext.sql("""SELECT date, kdo(date) as kdo_date, new_filename from train_set where date is not null""")
    train_val_df_ori.createOrReplaceTempView('train_val_df_ori')
    train_val_df = sqlContext.sql("""SELECT kdo_date as date, new_filename as filename from train_val_df_ori where kdo_date is not null and kdo_date > 1000 """)

    test_df_ori = sqlContext.sql("""SELECT date, kdo(date) as kdo_date, new_filename from test_set where date is not null""")
    test_df_ori.createOrReplaceTempView('test_df_ori')
    test_df = sqlContext.sql("""SELECT kdo_date as date, new_filename as filename from test_df_ori where kdo_date is not null and kdo_date > 1000 """)

    # Converting dataframes to Pandas
    p_test_df = test_df.toPandas()
    p_train_val_df = train_val_df.toPandas()

    # Splitting Train and Validation Set
    p_train_df = p_train_val_df.sample(frac=0.8, random_state=args.random_seed) #random state is a seed value
    p_val_df = p_train_val_df.drop(p_train_df.index)

    # Let's print some statistiscs
    print('TRAINING FIELDS & TYPES:  ')
    train_val_df.printSchema()

    print('\nTRAINING ENTRIES:  {}'.format(len(p_train_df)))
    print('VALIDATIO ENTRIES: {}'.format(len(p_val_df)))
    print('TEST ENTRIES:      {}'.format(len(p_test_df)))

    # Let's normalize the input dates
    train_date_mean = np.mean(p_train_df.date)
    train_date_std = np.std(p_train_df.date)
    print('\nTRAINING DATES MEAN: {:.4f}'.format(train_date_mean))
    print('TRAINING DATES STD:  {:.4f}'.format(train_date_std))

    p_train_df.date = (p_train_df.date-train_date_mean)/train_date_std
    p_val_df.date = (p_val_df.date-train_date_mean)/train_date_std
    p_test_df.date = (p_test_df.date-train_date_mean)/train_date_std

    ## Initializng data transformations
    manual_seed(args.random_seed)

    image_size = 224
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(int(1.1*image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    ## Initializng pyTorch dataloaders
    painter_dataset = {}
    painter_dataset['train'] = dataset.PaintersDataset(p_train_df, args.train_img_dir, data_transforms['train'], log_enabled=True)
    painter_dataset['val'] = dataset.PaintersDataset(p_val_df, args.train_img_dir, data_transforms['val'], log_enabled=True)
    painter_dataset['test'] = dataset.PaintersDataset(p_test_df, args.test_img_dir, data_transforms['val'], log_enabled=True)

    dataloaders = {}
    dataloaders['train'] = DataLoader(painter_dataset['train'], batch_size=args.batch_size, shuffle=True, num_workers=4)
    dataloaders['val'] = DataLoader(painter_dataset['val'], batch_size=args.batch_size, shuffle=True, num_workers=4)
    dataloaders['test'] = DataLoader(painter_dataset['test'], batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Let's initialize the device, in order to be able to train on GPU, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## Initializing the model, the loss and the optimizer
    if args.select_squeezenet:
        net = model.SqueezeNet_fc(1)
    else:
        net = model.Resnet152_fc(1)

    net.to(device)

    loss = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    # Setting the model path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Defining the training function
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25, resume_training=False, resuming_epoch=1):
        writer = SummaryWriter(args.logs_dir + args.experiment_name)

        # Let's manage situations in which we want to resume interrupted trainings
        if resume_training:
            model_filename = model_path + str(resuming_epoch - 1) + '.pth'
            checkpoint = torch.load(model_filename)
            model.load_state_dict(checkpoint['state_dict'])

        # Let's track training time for better planning resource usage
        since = time.time()

        # We will choose the best model, defined as the model with the smallest validation loss
        min_val_loss = 100000.0

        for epoch in range(num_epochs):
            if resume_training:
                if epoch < resuming_epoch:
                    continue

            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0

                # Iterate over data. Let's keep track of iteration and elapsed time
                iteration = 0
                t_0 = time.time()
                t_c = t_0

                for inputs, labels, filenames in dataloaders[phase]:
                    iteration = iteration + 1

                    inputs = inputs.to(device)
                    labels = labels.view(-1, 1).to(device)

                    # Set to zero all the optimizer gradients
                    optimizer.zero_grad()

                    # Forward Pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)

                        # Let's calculate the current loss and let's convert it in year error, for better understanding how the model is going
                        loss = criterion(outputs, labels)
                        err = loss * train_date_std

                        # Backpropagate if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Increment the running loss for generating final statistics
                    running_loss += loss.item() * inputs.size(0)

                    # Periodically print training status information
                    if iteration % 10 == 0:
                        print("{}: epoch: {: >4d} - {: >4d} out of {} ({: >2.2f}%) - loss: {:.4f} - err: {: >4.0f} years - cycle time: {:.4f} - elapsed time: {:.4f}".format(
                            phase, epoch, iteration, len(dataloaders[phase]), 100 * iteration / len(dataloaders[phase]),
                            loss.item(), err.item(), time.time() - t_c, time.time() - t_0))

                        t_c = time.time()

                # If in training, update the learning rate
                if phase == 'train':
                    scheduler.step()

                # Calculating epoch statistics and printing
                epoch_loss = running_loss / (args.batch_size*len(dataloaders[phase]))
                avg_err = epoch_loss * train_date_std
                print('{} Loss: {:.4f} Avg Error: {:.4f}'.format(phase, epoch_loss, avg_err))

                # Saving the model on each epoch
                print('Saving..')
                model_filename = model_path + str(epoch) + '.pth'
                torch.save({'state_dict': model.state_dict()}, model_filename)
                print('..done!')

                writer.add_scalar(phase + '/loss', epoch_loss, epoch)
                writer.add_scalar(phase + '/avg_err', avg_err, epoch)

                # Saving the best model, based on validation accuracy
                if phase == 'val' and epoch_loss < min_val_loss:
                    min_val_loss = epoch_loss

                    # model_filename = model_path + str(epoch) + '_best.pth'
                    # torch.save({'state_dict': model.state_dict()}, model_filename)

                    model_filename = model_path + 'best_model.pth'
                    torch.save({'state_dict': model.state_dict()}, model_filename)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # Load and return the trained model
        model_filename = model_path + 'best_model.pth'
        checkpoint = torch.load(model_filename)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    train_model(net, loss, optimizer, scheduler, num_epochs=args.num_epochs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--experiment_name', type=str, help='Experiment Name. It will be used for naming trained models and logs.', default='deployment')

    # output parameters
    parser.add_argument('--output_dir', type=str, help='Directory where the experiment results will be stored.', default='../models/')
    parser.add_argument('--logs_dir', type=str, help='Directory where the experiment logs will be stored by Tensorboard.', default='../runs/')

    # input parameters
    parser.add_argument('--csv_filepath', type=str, help='Path to the file listing the image filename, artistID, genre, style, date, title...', default='../data/all_data_info.csv')
    parser.add_argument('--train_img_dir', type=str, help='Directory where the training images are located.', default='../data/train/')
    parser.add_argument('--test_img_dir', type=str, help='Directory where the testing images are located.', default='../data/test/')

    # training parameters
    parser.add_argument('--select_squeezenet', help='Set to True if willing to use SqueezeNet instead of ResNet152.', type=utils.arg_str2bool, default=False)

    parser.add_argument('--batch_size', help='Batch Size.', type=int, default=16)
    parser.add_argument('--num_epochs', help='Number of training epochs.', type=int, default=50)
    parser.add_argument('--learning_rate', help='Learning Rate.', type=float, default=0.001)
    parser.add_argument('--momentum', help='Momentum.', type=float, default=0.9)
    parser.add_argument('--scheduler_step_size', help='Scheduler Step Size for adjusting learning rate.', type=int, default=10)
    parser.add_argument('--scheduler_gamma', help='Scheduler magnitude of learning rate adjustment.', type=float, default=0.1)

    parser.add_argument('--random_seed', help='Random Seed.', type=int, default=699)

    args = parser.parse_args()

    print(args)
    main(args)
