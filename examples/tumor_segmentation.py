import os
import torch
import numpy as np
import sys
import cv2
from torch.utils.data import DataLoader
import albumentations as albu
import pickle
import matplotlib.pyplot as plt
from brats_dataset import Dataset
from configs import configs

sys.path.insert(0, '/home/qasima/segmentation_models.pytorch')
import segmentation_models_pytorch as smp

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

"""
Using a U-net architecture for segmentation of Tumor Modalities
"""


class UnetTumorSegmentator:
    def __init__(self, mode, model_name, pure_ratio, synthetic_ratio, augmented_ratio):
        self.root_dir = '/home/qasima/segmentation_models.pytorch/'
        self.data_dir = '/home/qasima/segmentation_models.pytorch/data/'

        # epochs
        self.epochs_num = 100
        self.total_epochs = 0 + self.epochs_num
        self.continue_train = False

        # what proportion of pure data to be used
        self.pure_ratio = pure_ratio

        # proportion of synthetic or augmented data to be used, depending on the mode, w.r.t the pure dataset size
        # can be set upto 2.0 i.e. 200%
        self.synthetic_ratio = synthetic_ratio
        self.augmented_ratio = augmented_ratio

        # proportion of test data to be used
        self.test_ratio = 1.0

        # test and validation ratios to use
        self.validation_ratio = 0.1

        # training
        # mode = pure, none, elastic, coregistration, augmented, none_only or augmented_coregistered
        self.mode = mode
        self.encoder = 'resnet34'
        self.encoder_weights = None
        self.device = 'cuda'
        self.activation = 'softmax'
        self.loss = 'cross_entropy'

        self.all_classes = ['bg', 't_2', 't_1', 'b', 't_3']
        # classes to be trained upon
        self.classes = ['t_2', 't_1', 't_3']

        # paths
        self.model_name = model_name
        self.log_dir = self.root_dir + 'logs/' + self.loss + '/' + self.model_name
        self.model_dir = self.root_dir + '/models/' + self.loss + '/' + self.model_name
        self.result_dir = self.root_dir + '/results/' + self.loss + '/' + self.model_name

        # dataset paths
        self.x_dir = dict()
        self.y_dir = None
        self.x_dir_syn = dict()
        self.y_dir_syn = None
        self.x_dir_test = dict()
        self.y_dir_test = None

        # loaded or created model
        self.model = None

        # full dataset and the pure dataset
        self.full_dataset = None
        self.full_dataset_pure = None

        # model setup
        self.model_loss = None
        self.metrics = None
        self.optimizer = None

    def create_folders(self):
        # create folders for results and logs to be saved
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

    def set_dataset_paths(self):
        # pure dataset
        self.x_dir['t1ce'] = os.path.join(self.data_dir, 'train_t1ce_img_full')
        self.x_dir['flair'] = os.path.join(self.data_dir, 'train_flair_img_full')
        self.x_dir['t2'] = os.path.join(self.data_dir, 'train_t2_img_full')
        self.x_dir['t1'] = os.path.join(self.data_dir, 'train_t1_img_full')
        self.y_dir = os.path.join(self.data_dir, 'train_label_full')

        # set the synthetic dataset paths
        if self.mode == 'elastic':
            # elastic segmentation masks mode
            self.x_dir_syn['t1ce'] = os.path.join(self.data_dir, 'train_t1ce_img_full_elastic')
            self.x_dir_syn['flair'] = os.path.join(self.data_dir, 'train_flair_img_full_elastic')
            self.x_dir_syn['t2'] = os.path.join(self.data_dir, 'train_t2_img_full_elastic')
            self.x_dir_syn['t1'] = os.path.join(self.data_dir, 'train_t1_img_full_elastic')
            self.y_dir_syn = os.path.join(self.data_dir, 'train_label_full_elastic')

        elif self.mode == 'coregistration' or self.mode == 'augmented_coregistration':
            # coregistration or augmented_coregistration mode
            self.x_dir_syn['t1ce'] = os.path.join(self.data_dir, 'train_t1ce_img_full_coregistration')
            self.x_dir_syn['flair'] = os.path.join(self.data_dir, 'train_flair_img_full_coregistration')
            self.x_dir_syn['t2'] = os.path.join(self.data_dir, 'train_t2_img_full_coregistration')
            self.x_dir_syn['t1'] = os.path.join(self.data_dir, 'train_t1_img_full_coregistration')
            self.y_dir_syn = os.path.join(self.data_dir, 'train_label_full_coregistration')

        elif self.mode == 'none' or self.mode == 'none_only':
            # none mode or while training on none masks only
            self.x_dir_syn['t1ce'] = os.path.join(self.data_dir, 'train_t1ce_img_full_syn')
            self.x_dir_syn['flair'] = os.path.join(self.data_dir, 'train_flair_img_full_syn')
            self.x_dir_syn['t2'] = os.path.join(self.data_dir, 'train_t2_img_full_syn')
            self.x_dir_syn['t1'] = os.path.join(self.data_dir, 'train_t1_img_full_syn')
            self.y_dir_syn = os.path.join(self.data_dir, 'train_label_full_syn')

        # test dataset
        self.x_dir_test['t1ce'] = os.path.join(self.data_dir, 'train_t1ce_img_full_test')
        self.x_dir_test['flair'] = os.path.join(self.data_dir, 'train_flair_img_full_test')
        self.x_dir_test['t2'] = os.path.join(self.data_dir, 'train_t2_img_full_test')
        self.x_dir_test['t1'] = os.path.join(self.data_dir, 'train_t1_img_full_test')
        self.y_dir_test = os.path.join(self.data_dir, 'train_label_full_test')

    def create_dataset(self):
        # create the pure dataset
        self.full_dataset_pure = Dataset(
            self.x_dir,
            self.y_dir,
            classes=self.classes,
            augmentation=self.get_training_augmentation_padding(),
        )

        pure_size = int(len(self.full_dataset_pure) * self.pure_ratio)
        self.full_dataset_pure = torch.utils.data.Subset(self.full_dataset_pure, np.arange(pure_size))

        if self.mode == 'pure':
            # mode is pure then full dataset is the pure dataset
            self.full_dataset = self.full_dataset_pure

        elif self.mode == 'augmented':
            # mode is augmented so apply simple augmentations to pure dataset and concatenate with pure dataset
            full_dataset_augmented = Dataset(
                self.x_dir,
                self.y_dir,
                classes=self.classes,
                augmentation=self.get_training_augmentation_simple(),
            )
            augmented_size = int(len(self.full_dataset_pure) * self.augmented_ratio)
            full_dataset_augmented = torch.utils.data.Subset(full_dataset_augmented, np.arange(augmented_size))

            # 200%
            # full_dataset_augmented = torch.utils.data.ConcatDataset((full_dataset_augmented, full_dataset_augmented))
            self.full_dataset = torch.utils.data.ConcatDataset((self.full_dataset_pure, full_dataset_augmented))

        elif self.mode == 'augmented_coregistration':
            # mode is augmented_coregistration so the full dataset consists of augmented pure and coregistered dataset
            # along with the unaugmented ones
            full_dataset_augmented = Dataset(
                self.x_dir,
                self.y_dir,
                classes=self.classes,
                augmentation=self.get_training_augmentation_simple(),
            )
            augmented_size = int(len(self.full_dataset_pure) * self.augmented_ratio)
            full_dataset_augmented = torch.utils.data.Subset(full_dataset_augmented, np.arange(augmented_size))

            # 200
            # full_dataset_augmented = torch.utils.data.ConcatDataset((full_dataset_augmented, full_dataset_augmented))

            full_dataset_syn = Dataset(
                self.x_dir_syn,
                self.y_dir_syn,
                classes=self.classes,
                augmentation=self.get_training_augmentation_padding(),
            )

            synthetic_size = int(len(self.full_dataset_pure) * self.synthetic_ratio)
            full_dataset_syn = torch.utils.data.Subset(full_dataset_syn, np.arange(synthetic_size))

            full_dataset_syn_augmented = Dataset(
                self.x_dir_syn,
                self.y_dir_syn,
                classes=self.classes,
                augmentation=self.get_training_augmentation_simple(),
            )

            synthetic_size = int(len(self.full_dataset_pure) * self.synthetic_ratio)
            full_dataset_syn_augmented = torch.utils.data.Subset(full_dataset_syn_augmented, np.arange(synthetic_size))

            self.full_dataset = torch.utils.data.ConcatDataset((self.full_dataset_pure,
                                                                full_dataset_syn,
                                                                full_dataset_augmented,
                                                                full_dataset_syn_augmented))

        elif self.mode == 'none_only':
            # mode is none_only, full dataset consists of synthetic images generated from segmentation masks without
            # any augmentations
            full_dataset_syn = Dataset(
                self.x_dir_syn,
                self.y_dir_syn,
                classes=self.classes,
                augmentation=self.get_training_augmentation_padding(),
            )

            synthetic_size = int(len(self.full_dataset_pure) * self.synthetic_ratio)
            self.full_dataset = torch.utils.data.Subset(full_dataset_syn, np.arange(synthetic_size))
            self.full_dataset_pure = self.full_dataset

        else:
            # for modes elastic, coregistration and none simply add the corresponding synthetic images to pure dataset
            full_dataset_syn = Dataset(
                self.x_dir_syn,
                self.y_dir_syn,
                classes=self.classes,
                augmentation=self.get_training_augmentation_padding(),
            )

            synthetic_size = int(len(self.full_dataset_pure) * self.synthetic_ratio)
            full_dataset_syn = torch.utils.data.Subset(full_dataset_syn, np.arange(synthetic_size))

            # 200%
            # full_dataset_syn = torch.utils.data.ConcatDataset((full_dataset_syn, full_dataset_syn))
            self.full_dataset = torch.utils.data.ConcatDataset((self.full_dataset_pure, full_dataset_syn))

        return self.full_dataset, self.full_dataset_pure

    def create_model(self):
        # create or load the model
        if self.continue_train:
            self.model = torch.load(self.model_dir)
        else:
            self.model = smp.Unet(
                encoder_name=self.encoder,
                encoder_weights=self.encoder_weights,
                classes=len(self.classes),
                activation=self.activation,
            )
        return self.model

    def setup_model(self):
        # setup the model loss, metrics and optimizer
        self.model_loss = smp.utils.losses.BCEJaccardLoss(eps=1.)
        self.metrics = [
            smp.utils.metrics.IoUMetric(eps=1.),
            smp.utils.metrics.FscoreMetric(eps=1.),
        ]

        self.optimizer = torch.optim.Adam([
            {'params': self.model.decoder.parameters(), 'lr': 1e-4},
            {'params': self.model.encoder.parameters(), 'lr': 1e-6},
        ])

    @staticmethod
    def get_training_augmentation_padding():
        # Add padding to make image shape divisible by 32
        test_transform = [
            albu.PadIfNeeded(256, 256, cv2.BORDER_CONSTANT, (0, 0, 0))
        ]
        return albu.Compose(test_transform)

    @staticmethod
    def get_training_augmentation_simple():
        # simple augmentation to be applied during the training phase
        test_transform = [
            albu.Rotate(limit=15, interpolation=1, border_mode=4, value=None, always_apply=False, p=0.5),
            albu.VerticalFlip(always_apply=False, p=0.5),
            albu.HorizontalFlip(always_apply=False, p=0.5),
            albu.Transpose(always_apply=False, p=0.5),
            albu.CenterCrop(height=200, width=200, always_apply=False, p=0.5),
            albu.PadIfNeeded(256, 256, cv2.BORDER_CONSTANT, (0, 0, 0))
        ]
        return albu.Compose(test_transform)

    def load_results(self, model_name=None):
        # load the results
        if model_name is not None:
            log_dir = self.root_dir + 'logs/' + self.loss + '/' + model_name
        else:
            log_dir = self.log_dir
        with open(log_dir + '/train_loss', 'rb') as f:
            train_loss = pickle.load(f)
        with open(log_dir + '/valid_loss', 'rb') as f:
            valid_loss = pickle.load(f)
        with open(log_dir + '/train_score', 'rb') as f:
            train_score = pickle.load(f)
        with open(log_dir + '/valid_score', 'rb') as f:
            valid_score = pickle.load(f)
        return train_loss, valid_loss, train_score, valid_score

    def write_results(self, train_loss, valid_loss, train_score, valid_score):
        with open(self.log_dir + '/train_loss', 'wb') as f:
            pickle.dump(train_loss, f)
        with open(self.log_dir + '/valid_loss', 'wb') as f:
            pickle.dump(valid_loss, f)
        with open(self.log_dir + '/train_score', 'wb') as f:
            pickle.dump(train_score, f)
        with open(self.log_dir + '/valid_score', 'wb') as f:
            pickle.dump(valid_score, f)

    def train_model(self):
        # train the model
        train_epoch = smp.utils.train.TrainEpoch(
            self.model,
            loss=self.model_loss,
            metrics=self.metrics,
            optimizer=self.optimizer,
            device=self.device,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            self.model,
            loss=self.model_loss,
            metrics=self.metrics,
            device=self.device,
            verbose=True,
        )

        max_score = 0

        train_loss = np.zeros(self.epochs_num)
        valid_loss = np.zeros(self.epochs_num)
        train_score = np.zeros(self.epochs_num)
        valid_score = np.zeros(self.epochs_num)

        valid_size = int(self.validation_ratio * len(self.full_dataset_pure))
        remaining_size = len(self.full_dataset_pure) - valid_size

        for i in range(0, self.epochs_num):

            # During every epoch randomly sample from the dataset, for training and validation dataset members
            train_dataset = self.full_dataset
            valid_dataset, remaining_dataset = torch.utils.data.random_split(self.full_dataset_pure,
                                                                             [valid_size, remaining_size])
            train_loader = DataLoader(train_dataset, batch_size=36, shuffle=True, num_workers=8)
            valid_loader = DataLoader(valid_dataset, batch_size=18, drop_last=True)

            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou']:
                max_score = valid_logs['iou']
                torch.save(self.model, self.model_dir)
                print('Model saved!')

            if i == 10:
                self.optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')
            if i == 25:
                self.optimizer.param_groups[0]['lr'] = 1e-6
                print('Decrease decoder learning rate to 1e-6!')

            # get the loss logs
            train_loss[i] = train_logs['bce_jaccard_loss']
            valid_loss[i] = valid_logs['bce_jaccard_loss']
            train_score[i] = train_logs['f-score']
            valid_score[i] = valid_logs['f-score']

        # if continuing training, then load the previous loss and f-score logs
        if self.continue_train:
            train_loss_prev, valid_loss_prev, train_score_prev, valid_score_prev = self.load_results()
            train_loss = np.append(train_loss_prev, train_loss)
            valid_loss = np.append(valid_loss_prev, valid_loss)
            train_score = np.append(train_score_prev, train_score)
            valid_score = np.append(valid_score_prev, valid_score)
        self.write_results(train_loss, valid_loss, train_score, valid_score)

    def evaluate_model(self):
        # load best saved checkpoint
        best_model = torch.load(self.model_dir)

        full_dataset_test = Dataset(
            self.x_dir_test,
            self.y_dir_test,
            classes=self.classes,
            augmentation=self.get_training_augmentation_padding(),
        )

        # evaluate model on test set
        test_epoch = smp.utils.train.ValidEpoch(
            model=best_model,
            loss=self.model_loss,
            metrics=self.metrics,
            device=self.device,
        )

        test_size = int(len(full_dataset_test) * self.test_ratio)
        remaining_size = len(full_dataset_test) - test_size

        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset_test,
                                                                    [remaining_size, test_size])
        test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True, num_workers=1)
        test_epoch.run(test_loader)

    def plot_results(self, model_name=None):
        # load the results and make a plot
        if model_name is not None:
            plot_dir = self.root_dir + '/plots/' + self.loss + '/' + model_name + '.png'
        else:
            plot_dir = self.root_dir + '/plots/' + self.loss + '/' + self.model_name + '.png'

        x = np.arange(self.epochs_num)

        train_loss, valid_loss, train_score, valid_score = self.load_results(model_name)

        plt.plot(x, train_score)
        plt.plot(x, valid_score)
        plt.legend(['train_score', 'valid_score'], loc='lower right')
        plt.yticks(np.arange(0.0, 1.0, step=0.1))
        plt.savefig(plot_dir, bbox_inches='tight')
        plt.clf()

    @staticmethod
    def dice_coef(gt_mask, pr_mask):
        # the hard dice score implementation
        tp = np.sum(np.logical_and(pr_mask, gt_mask))
        sum_ = pr_mask.sum() + gt_mask.sum()
        if sum_ == 0:
            return 1.0
        dice = (2. * tp) / (pr_mask.sum() + gt_mask.sum())

        return dice

    def class_specific_dice(self, model_dir=None):
        # get class specific dice scores
        if model_dir is None:
            best_model = torch.load(self.model_dir)
        else:
            best_model = torch.load(model_dir)

        full_dataset_test = Dataset(
            self.x_dir_test,
            self.y_dir_test,
            classes=self.classes,
            augmentation=self.get_training_augmentation_padding(),
        )

        dice_avg = torch.zeros(len(self.classes))

        test_size = int(len(full_dataset_test) * self.test_ratio)
        remaining_size = len(full_dataset_test) - test_size

        full_dataset_test, train_dataset = torch.utils.data.random_split(full_dataset_test,
                                                                         [test_size, remaining_size])

        dice_avg_overall = 0
        dice_avg_tumor_core = 0

        test_loader = DataLoader(full_dataset_test, batch_size=3, shuffle=True, num_workers=1)

        for image, gt_mask in test_loader:
            image = image.to(self.device)
            gt_mask = gt_mask.cpu().detach().numpy()
            pr_mask = best_model.forward(image)

            activation_fn = torch.nn.Sigmoid()
            pr_mask = activation_fn(pr_mask)

            pr_mask = pr_mask.cpu().detach().numpy().round()

            for idx, cls in enumerate(self.classes):
                dice = self.dice_coef(gt_mask[:, idx, :, :], pr_mask[:, idx, :, :])
                dice_avg[idx] += dice

            dice = self.dice_coef(gt_mask, pr_mask)
            dice_avg_overall += dice
            gt_mask = gt_mask[:, [True, True, False], :, :]
            pr_mask = pr_mask[:, [True, True, False], :, :]
            dice = self.dice_coef(gt_mask, pr_mask)
            dice_avg_tumor_core += dice

        dice_avg /= (len(full_dataset_test) / 3)

        for idx, cls in enumerate(self.classes):
            print(self.classes[idx] + '{}'.format(dice_avg[idx]))

        print("Average : ", dice_avg_overall / (len(full_dataset_test) / 3))
        print("Average Tumor Core : ", dice_avg_tumor_core / (len(full_dataset_test) / 3), '\n')


if __name__ == "__main__":
    for config in configs:
        unet_model = UnetTumorSegmentator(**config)
        unet_model.create_folders()
        unet_model.set_dataset_paths()
        unet_model.create_dataset()
        unet_model.create_model()
        unet_model.setup_model()
        unet_model.train_model()
        unet_model.evaluate_model()
        unet_model.plot_results()
        unet_model.class_specific_dice(unet_model.classes)
