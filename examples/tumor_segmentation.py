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
MODEL_NAME = 'model_epochs_30_multi_modal_pure_resnet34_dice_1_pixel_syn'
EPOCHS_NUM = 15
SYNTHETIC_RATIO = 0.5
TEST_RATIO = 0.1

x_dir = dict()
x_dir_syn = dict()
x_dir_test = dict()
x_dir['t1ce'] = os.path.join(DATA_DIR, 'train_t1ce_img_full')
x_dir['t1'] = os.path.join(DATA_DIR, 'train_t1_img_full')
x_dir['t2'] = os.path.join(DATA_DIR, 'train_t2_img_full')
y_dir = os.path.join(DATA_DIR, 'train_label_full')
x_dir_syn['t1ce'] = os.path.join(DATA_DIR, 'train_t1ce_img_full_syn')
x_dir_syn['t1'] = os.path.join(DATA_DIR, 'train_t1_img_full_syn')
x_dir_syn['t2'] = os.path.join(DATA_DIR, 'train_t2_img_full_syn')
y_dir_syn = os.path.join(DATA_DIR, 'train_label_full_syn')
x_dir_test['t1ce'] = os.path.join(DATA_DIR, 'train_t1ce_img_full_syn')
x_dir_test['t1'] = os.path.join(DATA_DIR, 'train_t1_img_full_syn')
x_dir_test['t2'] = os.path.join(DATA_DIR, 'train_t2_img_full_syn')
y_dir_test = os.path.join(DATA_DIR, 'train_label_full_syn')


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
        self.ids = os.listdir(images_dir['t2'])
        self.images_fps_t1ce = [os.path.join(images_dir['t1ce'], image_id) for image_id in self.ids]
        self.images_fps_t1 = [os.path.join(images_dir['t1'], image_id) for image_id in self.ids]
        self.images_fps_t2 = [os.path.join(images_dir['t2'], image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data for all three modalities, i.e. t1ce, t1 and t2
        image_t1ce = cv2.imread(self.images_fps_t1ce[i], cv2.IMREAD_GRAYSCALE)
        image_t1 = cv2.imread(self.images_fps_t1[i], cv2.IMREAD_GRAYSCALE)
        image_t2 = cv2.imread(self.images_fps_t2[i], cv2.IMREAD_GRAYSCALE)
        image = np.stack([image_t1ce, image_t1, image_t2], axis=-1).astype('float32')
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)

        if image.shape[0] == 256:
            mask = cv2.copyMakeBorder(mask, 8, 8, 8, 8, cv2.BORDER_CONSTANT, (0, 0, 0))

        # extract certain classes from mask
        masks = [(mask == v) for v in self.class_values]
        mask_stacked = np.stack(masks, axis=-1).astype('float32')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_stacked)
            image, mask_stacked = sample['image'], sample['mask']

        # modification: adding a singleton dimension so that the image can be processed
        mask_stacked = np.swapaxes(mask_stacked, 1, 2)
        mask_stacked = np.swapaxes(mask_stacked, 0, 1)
        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 1, 2)

        return image, mask_stacked

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

model = torch.load('./' + MODEL_NAME)

'''
# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)
'''
full_dataset = Dataset(
    x_dir,
    y_dir,
    classes=CLASSES,
    augmentation=get_training_augmentation(),
)

full_dataset_syn = Dataset(
    x_dir_syn,
    y_dir_syn,
    classes=CLASSES,
    augmentation=get_training_augmentation(),
)

synthetic_size = len(full_dataset_syn) - int(len(full_dataset_syn)*(1-SYNTHETIC_RATIO))

full_dataset_syn = torch.utils.data.Subset(full_dataset_syn, np.arange(synthetic_size))
full_dataset = torch.utils.data.ConcatDataset((full_dataset, full_dataset_syn))

train_size = int(0.9 * len(full_dataset))
valid_size = len(full_dataset) - train_size

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

'''
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

    if i == 10:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
'''
# load best saved checkpoint
best_model = torch.load('./' + MODEL_NAME)

full_dataset_test = Dataset(
    x_dir_test,
    y_dir_test,
    classes=CLASSES,
    augmentation=get_training_augmentation(),
)

# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

test_size = int(len(full_dataset_test)*TEST_RATIO)
remaining_size = len(full_dataset_test) - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset_test,
                                                                 [remaining_size, test_size])
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=2)
logs = test_epoch.run(test_loader)

best_model = torch.load('./' + MODEL_NAME)
model_result_dir = os.path.join(RESULT_DIR, MODEL_NAME)

if not os.path.exists(model_result_dir):
    os.mkdir(model_result_dir)

# save some prediction mask samples
for i in range(10):
    n = np.random.choice(len(full_dataset))
    image, gt_mask = full_dataset[n]

    gt_mask = gt_mask.squeeze()

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = pr_mask.squeeze().cpu().numpy().round()
    gt_img = visualize(gt_mask)
    pr_img = visualize(pr_mask)
    image = np.squeeze(image)
    imsave(model_result_dir + '/real_image_t1ce_{}.png'.format(i), image[0, :, :])
    imsave(model_result_dir + '/real_image_t1_{}.png'.format(i), image[1, :, :])
    imsave(model_result_dir + '/real_image_t2_{}.png'.format(i), image[2, :, :])
    imsave(model_result_dir + '/real_mask_{}.png'.format(i), gt_img)
    imsave(model_result_dir + '/predicted_mask_{}.png'.format(i), pr_img)
