import os
import torch
import numpy as np
import sys
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
from scipy.misc import imsave

sys.path.insert(0, '/home/qasima/segmentation_models.pytorch')
import segmentation_models_pytorch as smp

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

DATA_DIR = '/home/qasima/segmentation_models.pytorch/data/'
RESULT_DIR = '/home/qasima/segmentation_models.pytorch/results/'
MODEL_NAME = 'model_epochs_10_pure_resnet34_CCE'
EPOCHS_NUM = 5

x_dir = os.path.join(DATA_DIR, 'train_t1ce_img_full')
x_dir_seg_csf = os.path.join(DATA_DIR, 'train_t1ce_img_full')
y_dir = os.path.join(DATA_DIR, 'train_label_full')


# this function fuses the predicted mask classes into one image
def visualize(image):
    sh = image.shape
    image_grayscale = np.zeros((sh[1], sh[2]))
    for j in range(0, sh[0]):
        image_grayscale = np.where(image[j] != 0, image[j] + j, image_grayscale)
    return image_grayscale * 255.0


class Dataset(BaseDataset):
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

        # extract certain classes from mask
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
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


# Add paddings to make image shape divisible by 32
def get_training_augmentation():
    test_transform = [
        albu.PadIfNeeded(256, 256, cv2.BORDER_CONSTANT, (0, 0, 0))
    ]
    return albu.Compose(test_transform)


# modification: changing from se_resnext50_32x4d to resnet34
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

# select only the tumor classes
CLASSES = ['t_2', 't_1', 't_3']
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
    augmentation=get_training_augmentation(),
)

train_size = int(0.9 * len(full_dataset))
valid_size = len(full_dataset) - train_size

loss = smp.utils.losses.BCEDiceLoss(eps=1.)
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


max_score = 0

for i in range(0, EPOCHS_NUM):

    # during every epoch randomly sample from the dataset, for training and validation dataset members
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset,
                                                                 [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=3, shuffle=False, num_workers=2)

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


best_model = torch.load('./' + MODEL_NAME)
model_result_dir = os.path.join(RESULT_DIR, MODEL_NAME)

# save some prediction mask samples
for i in range(5):
    n = np.random.choice(len(full_dataset))
    image, gt_mask = full_dataset[n]

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
