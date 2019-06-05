import os
import torch
import numpy as np
import sys

sys.path.insert(0, '/home/qasima/segmentation_models.pytorch')
import segmentation_models_pytorch as smp
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
from scipy.misc import imsave

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

DATA_DIR = '/home/qasima/segmentation_models.pytorch/data/'
RESULT_DIR = '/home/qasima/segmentation_models.pytorch/results/'
MODEL_NAME = 'model_epochs_30_pure_seg_csf_resnet34_50_percent'

if not os.path.exists(DATA_DIR):
    print('Loading data...')
    os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
    print('Done!')

x_dir = os.path.join(DATA_DIR, 'train_t1ce_img_train_full')
x_dir_seg_csf = os.path.join(DATA_DIR, 'train_t1ce_img_fused_csf_full')
y_dir = os.path.join(DATA_DIR, 'train_label_full')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')


def visualize(image):
    sh = image.shape
    image_grayscale = np.zeros((sh[1], sh[2]))
    for j in range(0, sh[0]):
        image_grayscale = np.where(image[j] != 0, image[j] + j, image_grayscale)
    return image_grayscale * 255.0


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """

    CLASSES = ['bg', 't_2', 't_1', 'b', 't_3']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_GRAYSCALE)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # modification: adding a singleton dimension so that the image can be processed
        image = image[None, :]
        mask = np.swapaxes(mask, 1, 2)
        mask = np.swapaxes(mask, 0, 1)
        image = image.astype('float32')
        mask = mask.astype('float32')

        return image, mask

    def __len__(self):
        return len(self.ids)


# Lets look at data we have
# dataset = Dataset(x_train_dir, y_train_dir, classes=['car'])


# ### Augmentations

# Data augmentation is a powerful technique to increase the amount of your data and prevent model overfitting.  
# If you not familiar with such trick read some of these articles:
#  - [The Effectiveness of Data Augmentation in Image Classification using Deep
# Learning](http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf)
#  - [Data Augmentation | How to use Deep Learning when you have Limited Data](https://medium.com/nanonets/how-to-use-
#  deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced)
#  - [Data Augmentation Experimentation](https://towardsdatascience.com/data-augmentation-experimentation-3e274504f04b)
# 
# Since our dataset is very small we will apply a large number of different augmentations:
#  - horizontal flip
#  - affine transforms
#  - perspective transforms
#  - brightness/contrast/colors manipulations
#  - image bluring and sharpening
#  - gaussian noise
#  - random crops
# 
# All this transforms can be easily applied with [**Albumentations**](https://github.com/albu/albumentations/) -
# fast augmentation library.
# For detailed explanation of image transformations you can look at [kaggle salt segmentation exmaple]
# (https://github.com/albu/albumentations/blob/master/notebooks/example_kaggle_salt.ipynb) provided by
# [**Albumentations**](https://github.com/albu/albumentations/) authors.

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_train_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(256, 256, cv2.BORDER_CONSTANT, (0, 0, 0))
    ]
    return albu.Compose(test_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# modification: changing from se_resnext50_32x4d to resnet34
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

CLASSES = ['bg', 't_2', 't_1', 'b', 't_3']
ACTIVATION = 'sigmoid'

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

full_dataset = Dataset(
    x_dir,
    y_dir,
    classes=CLASSES,
    augmentation=get_train_augmentation(),
)

full_dataset_seg_csf = Dataset(
    x_dir_seg_csf,
    y_dir,
    classes=CLASSES,
    augmentation=get_train_augmentation(),
)

train_size = int(0.8 * len(full_dataset))
valid_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - valid_size

train_size_seg_csf = int(0.8 * len(full_dataset_seg_csf))
remaining_size_seg_csf = len(full_dataset_seg_csf) - train_size_seg_csf

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(full_dataset,
                                                                           [train_size, valid_size, test_size])

train_dataset_seg_csf, remaining_dataset_seg_csf = torch.utils.data.random_split(full_dataset_seg_csf,
                                                                                 [train_size_seg_csf,
                                                                                  remaining_size_seg_csf])

final_train_data = torch.utils.data.ConcatDataset([train_dataset, train_dataset_seg_csf])

train_loader = DataLoader(final_train_data, batch_size=12, shuffle=True, num_workers=8)
valid_loader = DataLoader(valid_dataset, batch_size=3, shuffle=False, num_workers=2)
test_dataloader = DataLoader(test_dataset)

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = smp.utils.losses.DiceLoss(eps=1.)
metrics = [
    smp.utils.metrics.IoUMetric(eps=1.),
    smp.utils.metrics.FscoreMetric(eps=1.),
]

optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': 1e-4},
    {'params': model.encoder.parameters(), 'lr': 1e-6},
])

# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# train model for 2 epochs
'''
max_score = 0

for i in range(0, 30):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou']:
        max_score = valid_logs['iou']
        torch.save(model, './' + MODEL_NAME)
        print('Model saved!')

    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
'''
# ## Test best saved model

# load best saved checkpoint
best_model = torch.load('./' + MODEL_NAME)

# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(test_dataloader)

best_model = torch.load('./' + MODEL_NAME)
model_result_dir = os.path.join(RESULT_DIR, MODEL_NAME)

for i in range(5):
    n = np.random.choice(len(test_dataset))
    image, gt_mask = test_dataset[n]

    gt_mask = gt_mask.squeeze()

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = pr_mask.squeeze().cpu().numpy().round()
    gt_img = visualize(gt_mask)
    pr_img = visualize(pr_mask)
    image = np.squeeze(image)
    imsave(model_result_dir + '/real_image_{}.png'.format(i), image)
    imsave(model_result_dir + '/real_mask_{}.png'.format(i), gt_img)
    imsave(model_result_dir + '/predicted_mask_{}.png'.format(i), pr_img)
