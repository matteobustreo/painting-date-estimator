import torch
from torchvision import transforms

import argparse
import time
import matplotlib
import os
from PIL import Image

import model
import utils


def main(args):
    train_date_mean = 1874.7548
    train_date_std = 121.9153

    ## Initializing the model, the loss and the optimizer
    if args.select_squeezenet:
        net = model.SqueezeNet_fc(1)
    else:
        net = model.Resnet152_fc(1)

    # Loading the model weights
    checkpoint = torch.load(args.model_filename)
    net.load_state_dict(checkpoint['state_dict'])

    # Instantiating the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Defining image transformation
    image_size = 224
    img_transforms = transforms.Compose([
                        transforms.Resize(int(1.1 * image_size)),
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

    # Loading the painting to evaluate
    if os.path.isfile(args.image_filename):
        with open(args.image_filename, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        img = None

    if not img is None:
        img = img_transforms(img).to(device)
        img = img.unsqueeze(0)

        net.eval()
        t_0 = time.time()
        predicted_year = net(img) * train_date_std + train_date_mean

        print('\nThe estimated creation year of the painting is: {}'.format(int(predicted_year[0].item())))
        print('Computing time: {:.4f} seconds.\n'.format(time.time() - t_0))
    else:
        print('\nImage not found.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_filename', help='Path to the painting we want to estimate the creation date.', type=str, default='../images/test_image.jpg')
    parser.add_argument('--model_filename', help='Path to the file with the model weights.', type=str, default='../models_pretrained/best_resnet152.pth')
    parser.add_argument('--select_squeezenet', help='Set to True if willing to use SqueezeNet instead of ResNet152.', type=utils.arg_str2bool, default=False)

    args = parser.parse_args()

    main(args)
