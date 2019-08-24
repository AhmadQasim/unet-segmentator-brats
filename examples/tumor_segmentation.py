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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

DATA_DIR = '/home/qasima/segmentation_models.pytorch/data/'

# Epochs
EPOCHS_NUM = 100
TOTAL_EPOCHS = 0 + EPOCHS_NUM
CONTINUE_TRAIN = False

# Ratios
PURE_RATIO = 1.0
SYNTHETIC_RATIO = 1.0
TEST_RATIO = 1.0
VALIDATION_RATIO = 0.1
AUGMENTED_RATIO = 1.0

# Training
# mode = pure, none, elastic, coregistration or augmented
MODE = 'augmented_coregistration'
ENCODER = 'resnet34'
ENCODER_WEIGHTS = None
DEVICE = 'cuda'
ACTIVATION = 'softmax'
LOSS = 'cross_entropy'

ALL_CLASSES = ['bg', 't_2', 't_1', 'b', 't_3']
# classes to be used
CLASSES = ['t_2', 't_1', 't_3']

# Paths
MODEL_NAME = 'model_epochs100_precent100_augmented_vis'
LOG_DIR = '/home/qasima/segmentation_models.pytorch/logs/' + LOSS + '/' + MODEL_NAME
MODEL_DIR = '/home/qasima/segmentation_models.pytorch/models/' + LOSS + '/' + MODEL_NAME
RESULT_DIR = '/home/qasima/segmentation_models.pytorch/results/' + LOSS + '/' + MODEL_NAME

# Dataset paths
x_dir = dict()
x_dir_syn = dict()
x_dir_test = dict()
x_dir['t1ce'] = os.path.join(DATA_DIR, 'train_t1ce_img_full')
x_dir['flair'] = os.path.join(DATA_DIR, 'train_flair_img_full')
x_dir['t2'] = os.path.join(DATA_DIR, 'train_t2_img_full')
x_dir['t1'] = os.path.join(DATA_DIR, 'train_t1_img_full')
y_dir = os.path.join(DATA_DIR, 'train_label_full')

if MODE == 'elastic':
    x_dir_syn['t1ce'] = os.path.join(DATA_DIR, 'train_t1ce_img_full_elastic')
    x_dir_syn['flair'] = os.path.join(DATA_DIR, 'train_flair_img_full_elastic')
    x_dir_syn['t2'] = os.path.join(DATA_DIR, 'train_t2_img_full_elastic')
    x_dir_syn['t1'] = os.path.join(DATA_DIR, 'train_t1_img_full_elastic')
    y_dir_syn = os.path.join(DATA_DIR, 'train_label_full_elastic')
elif MODE == 'coregistration' or MODE == 'augmented_coregistration':
    x_dir_syn['t1ce'] = os.path.join(DATA_DIR, 'train_t1ce_img_full_coregistration')
    x_dir_syn['flair'] = os.path.join(DATA_DIR, 'train_flair_img_full_coregistration')
    x_dir_syn['t2'] = os.path.join(DATA_DIR, 'train_t2_img_full_coregistration')
    x_dir_syn['t1'] = os.path.join(DATA_DIR, 'train_t1_img_full_coregistration')
    y_dir_syn = os.path.join(DATA_DIR, 'train_label_full_coregistration')
elif MODE == 'none':
    x_dir_syn['t1ce'] = os.path.join(DATA_DIR, 'train_t1ce_img_full_syn')
    x_dir_syn['flair'] = os.path.join(DATA_DIR, 'train_flair_img_full_syn')
    x_dir_syn['t2'] = os.path.join(DATA_DIR, 'train_t2_img_full_syn')
    x_dir_syn['t1'] = os.path.join(DATA_DIR, 'train_t1_img_full_syn')
    y_dir_syn = os.path.join(DATA_DIR, 'train_label_full_syn')

x_dir_test['t1ce'] = os.path.join(DATA_DIR, 'train_t1ce_img_full_test')
x_dir_test['flair'] = os.path.join(DATA_DIR, 'train_flair_img_full_test')
x_dir_test['t2'] = os.path.join(DATA_DIR, 'train_t2_img_full_test')
x_dir_test['t1'] = os.path.join(DATA_DIR, 'train_t1_img_full_test')
y_dir_test = os.path.join(DATA_DIR, 'train_label_full_test')

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)


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


def get_training_augmentation_simple():
    test_transform = [
        # albu.RandomScale(scale_limit=0.1, interpolation=1, always_apply=False, p=0.5),
        albu.Rotate(limit=15, interpolation=1, border_mode=4, value=None, always_apply=False, p=0.5),
        albu.VerticalFlip(always_apply=False, p=0.5),
        albu.HorizontalFlip(always_apply=False, p=0.5),
        albu.Transpose(always_apply=False, p=0.5),
        albu.CenterCrop(height=200, width=200, always_apply=False, p=0.5),
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

    pure_size = int(len(full_dataset_pure) * PURE_RATIO)
    full_dataset_pure = torch.utils.data.Subset(full_dataset_pure, np.arange(pure_size))

    if MODE == 'pure':
        full_dataset = full_dataset_pure
    elif MODE == 'augmented':
        full_dataset_augmented = Dataset(
            x_dir,
            y_dir,
            classes=CLASSES,
            augmentation=get_training_augmentation_simple(),
        )
        augmented_size = int(len(full_dataset_augmented) * AUGMENTED_RATIO)
        full_dataset_augmented = torch.utils.data.Subset(full_dataset_augmented, np.arange(augmented_size))

        # 200%
        # full_dataset_augmented = torch.utils.data.ConcatDataset((full_dataset_augmented, full_dataset_augmented))
        full_dataset = torch.utils.data.ConcatDataset((full_dataset_pure, full_dataset_augmented))
    elif MODE == 'augmented_coregistration':
        full_dataset_augmented = Dataset(
            x_dir,
            y_dir,
            classes=CLASSES,
            augmentation=get_training_augmentation_simple(),
        )
        augmented_size = int(len(full_dataset_augmented) * AUGMENTED_RATIO)
        full_dataset_augmented = torch.utils.data.Subset(full_dataset_augmented, np.arange(augmented_size))
        full_dataset_augmented = torch.utils.data.ConcatDataset((full_dataset_augmented, full_dataset_augmented))

        full_dataset_syn = Dataset(
            x_dir_syn,
            y_dir_syn,
            classes=CLASSES,
            augmentation=get_training_augmentation(),
        )

        synthetic_size = int(len(full_dataset_syn) * SYNTHETIC_RATIO)
        full_dataset_syn = torch.utils.data.Subset(full_dataset_syn, np.arange(synthetic_size))

        full_dataset_syn_augmented = Dataset(
            x_dir_syn,
            y_dir_syn,
            classes=CLASSES,
            augmentation=get_training_augmentation_simple(),
        )

        synthetic_size = int(len(full_dataset_syn_augmented) * SYNTHETIC_RATIO)
        full_dataset_syn_augmented = torch.utils.data.Subset(full_dataset_syn_augmented, np.arange(synthetic_size))

        full_dataset = torch.utils.data.ConcatDataset((full_dataset_pure, full_dataset_syn, full_dataset_augmented,
                                                       full_dataset_syn_augmented))

    else:
        full_dataset_syn = Dataset(
            x_dir_syn,
            y_dir_syn,
            classes=CLASSES,
            augmentation=get_training_augmentation(),
        )

        synthetic_size = int(len(full_dataset_syn) * SYNTHETIC_RATIO)
        full_dataset_syn = torch.utils.data.Subset(full_dataset_syn, np.arange(synthetic_size))

        # 200%
        # full_dataset_syn = torch.utils.data.ConcatDataset((full_dataset_syn, full_dataset_syn))
        full_dataset = torch.utils.data.ConcatDataset((full_dataset_pure, full_dataset_syn))

    return full_dataset, full_dataset_pure


def load_results(model_name=None):
    if model_name is not None:
        log_dir = '/home/qasima/segmentation_models.pytorch/logs/' + LOSS + '/' + model_name
    else:
        log_dir = LOG_DIR
    with open(log_dir + '/train_loss', 'rb') as f:
        train_loss = pickle.load(f)
    with open(log_dir + '/valid_loss', 'rb') as f:
        valid_loss = pickle.load(f)
    with open(log_dir + '/train_score', 'rb') as f:
        train_score = pickle.load(f)
    with open(log_dir + '/valid_score', 'rb') as f:
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

loss = smp.utils.losses.BCEJaccardLoss(eps=1.)
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
        train_loader = DataLoader(train_dataset, batch_size=36, shuffle=True, num_workers=8)
        valid_loader = DataLoader(valid_dataset, batch_size=18, drop_last=True)

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou']:
            max_score = valid_logs['iou']
            torch.save(model, MODEL_DIR)
            print('Model saved!')

        if i == 10:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-6
            print('Decrease decoder learning rate to 1e-6!')
        train_loss[i] = train_logs['bce_jaccard_loss']
        valid_loss[i] = valid_logs['bce_jaccard_loss']
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
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True, num_workers=1)
    logs = test_epoch.run(test_loader)


def visualize_images():
    best_model = torch.load(MODEL_DIR)

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
        imsave(RESULT_DIR + '/real_image_t1ce_{}.png'.format(i), image[0, :, :])
        imsave(RESULT_DIR + '/real_image_t1_{}.png'.format(i), image[1, :, :])
        imsave(RESULT_DIR + '/real_image_t2_{}.png'.format(i), image[2, :, :])
        imsave(RESULT_DIR + '/real_mask_{}.png'.format(i), gt_img)
        imsave(RESULT_DIR + '/predicted_mask_{}.png'.format(i), pr_img)


def plot_results(model_name):
    plot_dir = '/home/qasima/segmentation_models.pytorch/plots/' + LOSS + '/' + model_name + '.png'

    x = np.arange(EPOCHS_NUM)

    train_loss, valid_loss, train_score, valid_score = load_results(model_name)

    plt.plot(x, train_score)
    plt.plot(x, valid_score)
    plt.legend(['train_score', 'valid_score'], loc='lower right')
    plt.yticks(np.arange(0.0, 1.0, step=0.1))
    plt.savefig(plot_dir, bbox_inches='tight')
    plt.clf()


def dice_coef(gt_mask, pr_mask):
    tp = np.sum(np.logical_and(pr_mask, gt_mask))
    sum_ = pr_mask.sum() + gt_mask.sum()
    if sum_ == 0:
        return 1.0
    dice = (2. * tp) / (pr_mask.sum() + gt_mask.sum())

    return dice


def class_specific_dice(classes, model_path):
    # load best saved checkpoint
    best_model = torch.load(model_path)

    full_dataset_test = Dataset(
        x_dir_test,
        y_dir_test,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
    )

    metric = smp.utils.metrics.FscoreMetric(eps=1.)
    dice_avg = torch.zeros(len(classes))

    test_size = int(len(full_dataset_test) * TEST_RATIO)
    remaining_size = len(full_dataset_test) - test_size

    full_dataset_test, train_dataset = torch.utils.data.random_split(full_dataset_test,
                                                      [test_size, remaining_size])

    dice_avg_overall = 0

    test_loader = DataLoader(full_dataset_test, batch_size=3, shuffle=True, num_workers=1)

    for image, gt_mask in test_loader:
        image = image.to(DEVICE)
        gt_mask = gt_mask.cpu().detach().numpy()
        # gt_mask = gt_mask.to(DEVICE)
        pr_mask = best_model.forward(image)

        activation_fn = torch.nn.Sigmoid()
        pr_mask = activation_fn(pr_mask)

        pr_mask = pr_mask.cpu().detach().numpy().round()

        for idx, cls in enumerate(classes):
             dice = dice_coef(gt_mask[:, idx, :, :], pr_mask[:, idx, :, :])
             dice_avg[idx] += dice

        dice = dice_coef(gt_mask, pr_mask)
        dice_avg_overall += dice

    dice_avg /= (len(full_dataset_test)/3)
    # print(dice_avg)

    for idx, cls in enumerate(classes):
        print(classes[idx] + '{}'.format(dice_avg[idx]))

    print("Average : ", dice_avg_overall/(len(full_dataset_test)/3), '\n')


for filename in os.listdir('/home/qasima/segmentation_models.pytorch/models/' + LOSS + '/'):
    print(filename)
    # class_specific_dice(['Core: ', 'Enhancing: ', 'Edema: '], '/home/qasima/segmentation_models.pytorch/models/cross_'
    #                                                          'entropy/' + filename)
    plot_results(filename)
    # evaluate_model()

