import os
import torch
import numpy as np
import sys
import cv2
from torch.utils.data import DataLoader
import albumentations as albu
from scipy.misc import imsave
import pickle
import matplotlib.pyplot as plt
from brats_dataset import Dataset

sys.path.insert(0, '/home/qasima/segmentation_models.pytorch')
import segmentation_models_pytorch as smp

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

DATA_DIR = '/home/qasima/segmentation_models.pytorch/data/'
MODEL_NAME = 'model_epochs30_precent200_vis'
LOG_DIR = '/home/qasima/segmentation_models.pytorch/logs/' + MODEL_NAME
PLOT_DIR = '/home/qasima/segmentation_models.pytorch/plots/' + MODEL_NAME + '.png'
MODEL_DIR = '/home/qasima/segmentation_models.pytorch/models/cross_entopy/' + MODEL_NAME
RESULT_DIR = '/home/qasima/segmentation_models.pytorch/results/' + MODEL_NAME
# total: 100
EPOCHS_NUM = 100
PURE_RATIO = 1.0
SYNTHETIC_RATIO = 1.0
MODE = 'elastic'
TEST_RATIO = 1.0
VALIDATION_RATIO = 0.1
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'
CLASSES = ['t_2', 't_1', 't_3']
ACTIVATION = 'softmax'
CONTINUE_TRAIN = True

x_dir = dict()
x_dir_syn = dict()
x_dir_test = dict()
x_dir['t1ce'] = os.path.join(DATA_DIR, 'train_t1ce_img_full')
x_dir['t1'] = os.path.join(DATA_DIR, 'train_t1_img_full')
x_dir['t2'] = os.path.join(DATA_DIR, 'train_t2_img_full')
y_dir = os.path.join(DATA_DIR, 'train_label_full')

if MODE == 'elastic':
    x_dir_syn['t1ce'] = os.path.join(DATA_DIR, 'train_t1ce_img_elastic_full_syn')
    x_dir_syn['t1'] = os.path.join(DATA_DIR, 'train_t1_img_elastic_full_syn')
    x_dir_syn['t2'] = os.path.join(DATA_DIR, 'train_t2_img_elastic_full_syn')
    y_dir_syn = os.path.join(DATA_DIR, 'train_label_elastic_full')
elif MODE == 'coregistration':
    x_dir_syn['t1ce'] = os.path.join(DATA_DIR, 'train_t1ce_img_full_coregistration')
    x_dir_syn['t1'] = os.path.join(DATA_DIR, 'train_t1_img_full_coregistration')
    x_dir_syn['t2'] = os.path.join(DATA_DIR, 'train_t2_img_full_coregistration')
    y_dir_syn = os.path.join(DATA_DIR, 'train_label_full_coregistration')
elif MODE == 'none':
    x_dir_syn['t1ce'] = os.path.join(DATA_DIR, 'train_t1ce_img_full_syn')
    x_dir_syn['t1'] = os.path.join(DATA_DIR, 'train_t1_img_full_syn')
    x_dir_syn['t2'] = os.path.join(DATA_DIR, 'train_t2_img_full_syn')
    y_dir_syn = os.path.join(DATA_DIR, 'train_label_full_syn')

x_dir_test['t1ce'] = os.path.join(DATA_DIR, 'train_t1ce_img_full_test')
x_dir_test['t1'] = os.path.join(DATA_DIR, 'train_t1_img_full_test')
x_dir_test['t2'] = os.path.join(DATA_DIR, 'train_t2_img_full_test')
y_dir_test = os.path.join(DATA_DIR, 'train_label_full_test')

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)


# this function fuses the predicted mask classes into one image
def visualize(image):
    sh = image.shape
    image_grayscale = np.zeros((sh[1], sh[2]))
    for j in range(0, sh[0]):
        image_grayscale = np.where(image[j] != 0, image[j] + j, image_grayscale)
    return image_grayscale * 255.0


# Add paddings to make image shape divisible by 32
def get_training_augmentation():
    test_transform = [
        albu.PadIfNeeded(256, 256, cv2.BORDER_CONSTANT, (0, 0, 0))
    ]
    return albu.Compose(test_transform)


def create_model():
    if CONTINUE_TRAIN:
        model_loaded = torch.load(MODEL_DIR)
    else:
        model_loaded = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )
    return model_loaded


def create_dataset():
    full_dataset_pure = Dataset(
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

    pure_size = int(len(full_dataset_pure) * PURE_RATIO)

    synthetic_size = int(len(full_dataset_syn) * SYNTHETIC_RATIO)

    full_dataset_pure = torch.utils.data.Subset(full_dataset_pure, np.arange(pure_size))
    full_dataset_syn = torch.utils.data.Subset(full_dataset_syn, np.arange(synthetic_size))

    # 200%
    full_dataset_syn = torch.utils.data.ConcatDataset((full_dataset_syn, full_dataset_syn))

    full_dataset = torch.utils.data.ConcatDataset((full_dataset_pure, full_dataset_syn))

    return full_dataset, full_dataset_pure


def load_results():
    with open(LOG_DIR + '/train_loss', 'rb') as f:
        train_loss = pickle.load(f)
    with open(LOG_DIR + '/valid_loss', 'rb') as f:
        valid_loss = pickle.load(f)
    with open(LOG_DIR + '/train_score', 'rb') as f:
        train_score = pickle.load(f)
    with open(LOG_DIR + '/valid_score', 'rb') as f:
        valid_score = pickle.load(f)
    return train_loss, valid_loss, train_score, valid_score


def write_results(train_loss, valid_loss, train_score, valid_score):
    with open(LOG_DIR + '/train_loss', 'wb') as f:
        pickle.dump(train_loss, f)
    with open(LOG_DIR + '/valid_loss', 'wb') as f:
        pickle.dump(valid_loss, f)
    with open(LOG_DIR + '/train_score', 'wb') as f:
        pickle.dump(train_score, f)
    with open(LOG_DIR + '/valid_score', 'wb') as f:
        pickle.dump(valid_score, f)


model = create_model()
full_dataset, full_dataset_pure = create_dataset()

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


def train_model():
    max_score = 0

    train_loss = np.zeros(EPOCHS_NUM)
    valid_loss = np.zeros(EPOCHS_NUM)
    train_score = np.zeros(EPOCHS_NUM)
    valid_score = np.zeros(EPOCHS_NUM)

    valid_size = int(VALIDATION_RATIO * len(full_dataset_pure))
    remaining_size = len(full_dataset_pure) - valid_size

    for i in range(0, EPOCHS_NUM):

        # during every epoch randomly sample from the dataset, for training and validation dataset members
        train_dataset = full_dataset
        valid_dataset, remaining_dataset = torch.utils.data.random_split(full_dataset_pure, [valid_size, remaining_size])
        train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=8)
        valid_loader = DataLoader(valid_dataset, batch_size=3, drop_last=True)

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou']:
            max_score = valid_logs['iou']
            torch.save(MODEL_DIR)
            print('Model saved!')

        if i == 10:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-6
            print('Decrease decoder learning rate to 1e-6!')
        train_loss[i] = train_logs['dice_loss']
        valid_loss[i] = valid_logs['dice_loss']
        train_score[i] = train_logs['f-score']
        valid_score[i] = valid_logs['f-score']

    if CONTINUE_TRAIN:
        train_loss_prev, valid_loss_prev, train_score_prev, valid_score_prev = load_results()
        train_loss = np.append(train_loss_prev, train_loss)
        valid_loss = np.append(valid_loss_prev, valid_loss)
        train_score = np.append(train_score_prev, train_score)
        valid_score = np.append(valid_score_prev, valid_score)
    write_results(train_loss, valid_loss, train_score, valid_score)


def evaluate_model():
    # load best saved checkpoint
    best_model = torch.load(MODEL_DIR)

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

    test_size = int(len(full_dataset_test) * TEST_RATIO)
    remaining_size = len(full_dataset_test) - test_size

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset_test,
                                                                [remaining_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=2)
    logs = test_epoch.run(test_loader)


def visualize_images():
    best_model = torch.load(MODEL_DIR)
    model_result_dir = os.path.join(RESULT_DIR)
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


def plot_results():
    x = np.arange(EPOCHS_NUM)

    train_loss, valid_loss, train_score, valid_score = load_results()

    plt.plot(x, train_score)
    plt.plot(x, valid_score)
    plt.legend(['train_score', 'valid_score'], loc='lower right')
    plt.savefig(PLOT_DIR, bbox_inches='tight')


# train_model()
plot_results()
# evaluate_model()
