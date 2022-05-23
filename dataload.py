from __future__ import division
import os
from scipy import io as scio
import copy
from PIL.Image import Resampling
import sys
import random
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, ColorJitter, ToTensor, Normalize


def get_citypersons(root_dir='data/cityperson', mode='train'):
    all_img_path = os.path.join(root_dir, 'images')
    all_anno_path = os.path.join(root_dir, 'annotations')
    rows, cols = 1024, 2048

    anno_path = os.path.join(all_anno_path, 'anno_' + mode + '.mat')
    image_data = []
    annos = scio.loadmat(anno_path)
    index = 'anno_' + mode + '_aligned'
    valid_count = 0
    iggt_count = 0
    box_count = 0

    for l in range(len(annos[index][0])):
        anno = annos[index][0][l]
        cityname = anno[0][0][0][0]
        imgname = anno[0][0][1][0]
        gts = anno[0][0][2]
        img_path = os.path.join(all_img_path, mode + '/' + cityname + '/' + imgname)
        boxes = []
        ig_boxes = []
        vis_boxes = []
        for i in range(len(gts)):
            label, x1, y1, w, h = gts[i, :5]
            x1, y1 = max(int(x1), 0), max(int(y1), 0)
            w, h = min(int(w), cols - x1 - 1), min(int(h), rows - y1 - 1)
            xv1, yv1, wv, hv = gts[i, 6:]
            xv1, yv1 = max(int(xv1), 0), max(int(yv1), 0)
            wv, hv = min(int(wv), cols - xv1 - 1), min(int(hv), rows - yv1 - 1)

            if label == 1 and h >= 50:
                box = np.array([int(x1), int(y1), int(x1) + int(w), int(y1) + int(h)])
                boxes.append(box)
                vis_box = np.array([int(xv1), int(yv1), int(xv1) + int(wv), int(yv1) + int(hv)])
                vis_boxes.append(vis_box)
            else:
                ig_box = np.array([int(x1), int(y1), int(x1) + int(w), int(y1) + int(h)])
                ig_boxes.append(ig_box)
        boxes = np.array(boxes)
        vis_boxes = np.array(vis_boxes)
        ig_boxes = np.array(ig_boxes)

        if len(boxes) > 0:
            valid_count += 1
        annotation = {}
        annotation['filepath'] = img_path
        box_count += len(boxes)
        iggt_count += len(ig_boxes)
        annotation['bboxes'] = boxes
        annotation['vis_bboxes'] = vis_boxes
        annotation['ignoreareas'] = ig_boxes
        image_data.append(annotation)

    return image_data


def _brightness(image, min=0.5, max=2.0):
    '''
    Randomly change the brightness of the input image.
    Protected against overflow.
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    random_br = np.random.uniform(min, max)

    # To protect against overflow: Calculate a mask for all pixels
    # where adjustment of the brightness would exceed the maximum
    # brightness value and set the value to the maximum at those pixels.
    mask = hsv[:, :, 2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:, :, 2] * random_br)
    hsv[:, :, 2] = v_channel

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def resize_image(image, gts, igs, scale=(0.4, 1.5)):
    height, width = image.shape[0:2]
    ratio = np.random.uniform(scale[0], scale[1])
    # if len(gts)>0 and np.max(gts[:,3]-gts[:,1])>300:
    #     ratio = np.random.uniform(scale[0], 1.0)
    new_height, new_width = int(ratio * height), int(ratio * width)
    image = cv2.resize(image, (new_width, new_height))
    if len(gts) > 0:
        gts = np.asarray(gts, dtype=float)
        gts[:, 0:4:2] *= ratio
        gts[:, 1:4:2] *= ratio

    if len(igs) > 0:
        igs = np.asarray(igs, dtype=float)
        igs[:, 0:4:2] *= ratio
        igs[:, 1:4:2] *= ratio

    return image, gts, igs


def random_crop(image, gts, igs, crop_size, limit=8):
    img_height, img_width = image.shape[0:2]
    crop_h, crop_w = crop_size

    if len(gts) > 0:
        sel_id = np.random.randint(0, len(gts))
        sel_center_x = int((gts[sel_id, 0] + gts[sel_id, 2]) / 2.0)
        sel_center_y = int((gts[sel_id, 1] + gts[sel_id, 3]) / 2.0)
    else:
        sel_center_x = int(np.random.randint(0, img_width - crop_w + 1) + crop_w * 0.5)
        sel_center_y = int(np.random.randint(0, img_height - crop_h + 1) + crop_h * 0.5)

    crop_x1 = max(sel_center_x - int(crop_w * 0.5), int(0))
    crop_y1 = max(sel_center_y - int(crop_h * 0.5), int(0))
    diff_x = max(crop_x1 + crop_w - img_width, int(0))
    crop_x1 -= diff_x
    diff_y = max(crop_y1 + crop_h - img_height, int(0))
    crop_y1 -= diff_y
    cropped_image = np.copy(image[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w])
    # crop detections
    if len(igs) > 0:
        igs[:, 0:4:2] -= crop_x1
        igs[:, 1:4:2] -= crop_y1
        igs[:, 0:4:2] = np.clip(igs[:, 0:4:2], 0, crop_w)
        igs[:, 1:4:2] = np.clip(igs[:, 1:4:2], 0, crop_h)
        keep_inds = ((igs[:, 2] - igs[:, 0]) >= 8) & \
                    ((igs[:, 3] - igs[:, 1]) >= 8)
        igs = igs[keep_inds]
    if len(gts) > 0:
        ori_gts = np.copy(gts)
        gts[:, 0:4:2] -= crop_x1
        gts[:, 1:4:2] -= crop_y1
        gts[:, 0:4:2] = np.clip(gts[:, 0:4:2], 0, crop_w)
        gts[:, 1:4:2] = np.clip(gts[:, 1:4:2], 0, crop_h)

        before_area = (ori_gts[:, 2] - ori_gts[:, 0]) * (ori_gts[:, 3] - ori_gts[:, 1])
        after_area = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])

        keep_inds = ((gts[:, 2] - gts[:, 0]) >= limit) & \
                    (after_area >= 0.5 * before_area)
        gts = gts[keep_inds]

    return cropped_image, gts, igs


def random_pave(image, gts, igs, pave_size, limit=8):
    img_height, img_width = image.shape[0:2]
    pave_h, pave_w = pave_size
    # paved_image = np.zeros((pave_h, pave_w, 3), dtype=image.dtype)
    paved_image = np.ones((pave_h, pave_w, 3), dtype=image.dtype) * np.mean(image, dtype=int)
    pave_x = int(np.random.randint(0, pave_w - img_width + 1))
    pave_y = int(np.random.randint(0, pave_h - img_height + 1))
    paved_image[pave_y:pave_y + img_height, pave_x:pave_x + img_width] = image
    # pave detections
    if len(igs) > 0:
        igs[:, 0:4:2] += pave_x
        igs[:, 1:4:2] += pave_y
        keep_inds = ((igs[:, 2] - igs[:, 0]) >= 8) & \
                    ((igs[:, 3] - igs[:, 1]) >= 8)
        igs = igs[keep_inds]

    if len(gts) > 0:
        gts[:, 0:4:2] += pave_x
        gts[:, 1:4:2] += pave_y
        keep_inds = ((gts[:, 2] - gts[:, 0]) >= limit)
        gts = gts[keep_inds]

    return paved_image, gts, igs


def augment(img_data, c, img):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    img_data_aug = copy.deepcopy(img_data)
    if img is None:
        img = cv2.imread(img_data_aug['filepath'])
    img_height, img_width = img.shape[:2]

    # random brightness
    if c.brightness and np.random.randint(0, 2) == 0:
        img = _brightness(img, min=c.brightness[0], max=c.brightness[1])
    # random horizontal flip
    if c.use_horizontal_flips and np.random.randint(0, 2) == 0:
        img = cv2.flip(img, 1)
        if len(img_data_aug['bboxes']) > 0:
            img_data_aug['bboxes'][:, [0, 2]] = img_width - img_data_aug['bboxes'][:, [2, 0]]
        if len(img_data_aug['ignoreareas']) > 0:
            img_data_aug['ignoreareas'][:, [0, 2]] = img_width - img_data_aug['ignoreareas'][:, [2, 0]]

    gts = np.copy(img_data_aug['bboxes'])
    igs = np.copy(img_data_aug['ignoreareas'])

    img, gts, igs = resize_image(img, gts, igs, scale=(0.4, 1.5))
    if img.shape[0] >= c.size_train[0]:
        img, gts, igs = random_crop(img, gts, igs, c.size_train, limit=16)
    else:
        img, gts, igs = random_pave(img, gts, igs, c.size_train, limit=16)

    img_data_aug['bboxes'] = gts
    img_data_aug['ignoreareas'] = igs

    img_data_aug['width'] = c.size_train[1]
    img_data_aug['height'] = c.size_train[0]

    return img_data_aug, img


class CityPersons(Dataset):
    def __init__(self, path, mode, config, transform=None):
        if transform is None:
            transform = transforms.Compose(
                [transforms.ColorJitter(brightness=0.5), transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.dataset = get_citypersons(root_dir=path, mode=mode)
        self.dataset_len = len(self.dataset)
        self.mode = mode

        if self.mode == 'train' and config.train_random:
            random.shuffle(self.dataset)
        self.config = config
        self.transform = transform

        if self.mode == 'train':
            self.preprocess = RandomResizeFix(size=config.size_train, scale=(0.4, 1.5))
        else:
            self.preprocess = None

    def __getitem__(self, item):

        # input is RGB order, and normalized
        img_data = self.dataset[item]
        img = Image.open(img_data['filepath'])

        if self.mode == 'train':
            gts = img_data['bboxes'].copy()
            igs = img_data['ignoreareas'].copy()

            x_img, gts, igs = self.preprocess(img, gts, igs)

            y_center, y_height, y_offset = self.calc_gt_center(gts, igs, radius=2, stride=self.config.down)

            if self.transform is not None:
                x_img = self.transform(x_img)

            return x_img, [y_center, y_height, y_offset]

        else:
            if self.transform is not None:
                x_img = self.transform(img)
            else:
                x_img = img

            return x_img

    def __len__(self):
        return self.dataset_len

    def calc_gt_center(self, gts, igs, radius=2, stride=4):

        def gaussian(kernel):
            sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
            s = 2 * (sigma ** 2)
            dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
            return np.reshape(dx, (-1, 1))

        scale_map = np.zeros((2, int(self.config.size_train[0] / stride), int(self.config.size_train[1] / stride)))
        offset_map = np.zeros((3, int(self.config.size_train[0] / stride), int(self.config.size_train[1] / stride)))
        pos_map = np.zeros((3, int(self.config.size_train[0] / stride), int(self.config.size_train[1] / stride)))
        pos_map[1, :, :, ] = 1  # channel 1: 1-value mask, ignore area will be set to 0

        if len(igs) > 0:
            igs = igs / stride
            for ind in range(len(igs)):
                x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(
                    np.ceil(igs[ind, 3]))
                pos_map[1, y1:y2, x1:x2] = 0

        if len(gts) > 0:
            gts = gts / stride
            for ind in range(len(gts)):
                x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(
                    gts[ind, 3])
                c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)

                dx = gaussian(x2 - x1)
                dy = gaussian(y2 - y1)
                gau_map = np.multiply(dy, np.transpose(dx))

                pos_map[0, y1:y2, x1:x2] = np.maximum(pos_map[0, y1:y2, x1:x2], gau_map)  # gauss map
                pos_map[1, y1:y2, x1:x2] = 1  # 1-mask map
                pos_map[2, c_y, c_x] = 1  # center map

                scale_map[0, c_y - radius:c_y + radius + 1, c_x - radius:c_x + radius + 1] = np.log(
                    gts[ind, 3] - gts[ind, 1])  # log value of height
                scale_map[1, c_y - radius:c_y + radius + 1, c_x - radius:c_x + radius + 1] = 1  # 1-mask

                offset_map[0, c_y, c_x] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5  # height-Y offset
                offset_map[1, c_y, c_x] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5  # width-X offset
                offset_map[2, c_y, c_x] = 1  # 1-mask

        return pos_map, scale_map, offset_map


class RandomResizeFix(object):
    """
    Args:
        size: expected output size of each edge
        scale: scale factor
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.4, 1.5), interpolation=Resampling.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.scale = scale

    def __call__(self, img, gts, igs):
        # resize image
        w, h = img.size
        ratio = np.random.uniform(self.scale[0], self.scale[1])
        n_w, n_h = int(ratio * w), int(ratio * h)
        img = img.resize((n_w, n_h), self.interpolation)
        gts = gts.copy()
        igs = igs.copy()

        # resize label
        if len(gts) > 0:
            gts = np.asarray(gts, dtype=float)
            gts *= ratio

        if len(igs) > 0:
            igs = np.asarray(igs, dtype=float)
            igs *= ratio

        # random flip
        w, h = img.size
        if np.random.randint(0, 2) == 0:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if len(gts) > 0:
                gts[:, [0, 2]] = w - gts[:, [2, 0]]
            if len(igs) > 0:
                igs[:, [0, 2]] = w - igs[:, [2, 0]]

        if h >= self.size[0]:
            # random crop
            img, gts, igs = self.random_crop(img, gts, igs, self.size, limit=16)
        else:
            # random pad
            img, gts, igs = self.random_pave(img, gts, igs, self.size, limit=16)

        return img, gts, igs

    @staticmethod
    def random_crop(img, gts, igs, size, limit=8):
        w, h = img.size
        crop_h, crop_w = size

        if len(gts) > 0:
            sel_id = np.random.randint(0, len(gts))
            sel_center_x = int((gts[sel_id, 0] + gts[sel_id, 2]) / 2.0)
            sel_center_y = int((gts[sel_id, 1] + gts[sel_id, 3]) / 2.0)
        else:
            sel_center_x = int(np.random.randint(0, w - crop_w + 1) + crop_w * 0.5)
            sel_center_y = int(np.random.randint(0, h - crop_h + 1) + crop_h * 0.5)

        crop_x1 = max(sel_center_x - int(crop_w * 0.5), int(0))
        crop_y1 = max(sel_center_y - int(crop_h * 0.5), int(0))
        diff_x = max(crop_x1 + crop_w - w, int(0))
        crop_x1 -= diff_x
        diff_y = max(crop_y1 + crop_h - h, int(0))
        crop_y1 -= diff_y
        cropped_img = img.crop((crop_x1, crop_y1, crop_x1 + crop_w, crop_y1 + crop_h))

        # crop detections
        if len(igs) > 0:
            igs[:, 0:4:2] -= crop_x1
            igs[:, 1:4:2] -= crop_y1
            igs[:, 0:4:2] = np.clip(igs[:, 0:4:2], 0, crop_w)
            igs[:, 1:4:2] = np.clip(igs[:, 1:4:2], 0, crop_h)
            keep_inds = ((igs[:, 2] - igs[:, 0]) >= 8) & ((igs[:, 3] - igs[:, 1]) >= 8)
            igs = igs[keep_inds]

        if len(gts) > 0:
            before_area = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
            gts[:, 0:4:2] -= crop_x1
            gts[:, 1:4:2] -= crop_y1
            gts[:, 0:4:2] = np.clip(gts[:, 0:4:2], 0, crop_w)
            gts[:, 1:4:2] = np.clip(gts[:, 1:4:2], 0, crop_h)

            after_area = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])

            keep_inds = ((gts[:, 2] - gts[:, 0]) >= limit) & (after_area >= 0.5 * before_area)
            gts = gts[keep_inds]

        return cropped_img, gts, igs

    @staticmethod
    def random_pave(img, gts, igs, size, limit=8):
        img = np.asarray(img)
        h, w = img.shape[0:2]
        pave_h, pave_w = size
        # paved_image = np.zeros((pave_h, pave_w, 3), dtype=image.dtype)
        paved_image = np.ones((pave_h, pave_w, 3), dtype=img.dtype) * np.mean(img, dtype=int)
        pave_x = int(np.random.randint(0, pave_w - w + 1))
        pave_y = int(np.random.randint(0, pave_h - h + 1))
        paved_image[pave_y:pave_y + h, pave_x:pave_x + w] = img
        # pave detections
        if len(igs) > 0:
            igs[:, 0:4:2] += pave_x
            igs[:, 1:4:2] += pave_y
            keep_inds = ((igs[:, 2] - igs[:, 0]) >= 8) & ((igs[:, 3] - igs[:, 1]) >= 8)
            igs = igs[keep_inds]

        if len(gts) > 0:
            gts[:, 0:4:2] += pave_x
            gts[:, 1:4:2] += pave_y
            keep_inds = ((gts[:, 2] - gts[:, 0]) >= limit)
            gts = gts[keep_inds]

        return Image.fromarray(paved_image), gts, igs


class Config(object):
    def __init__(self):
        self.gpu_ids = [0, 1]
        self.onegpu = 4
        self.num_epochs = 150
        self.add_epoch = 0
        self.iter_per_epoch = 2000
        self.init_lr = 2e-4
        self.alpha = 0.999

        # dataset
        self.train_path = './data/citypersons'
        self.train_random = True

        # setting for network architechture
        self.network = 'resnet50'  # or 'mobilenet'
        self.point = 'center'  # or 'top', 'bottom
        self.scale = 'h'  # or 'w', 'hw'
        self.num_scale = 1  # 1 for height (or width) prediction, 2 for height+width prediction
        self.offset = True  # append offset prediction or not
        self.down = 4  # downsampling rate of the feature map for detection
        self.radius = 2  # surrounding areas of positives for the scale map

        # setting for data augmentation
        self.use_horizontal_flips = True
        self.brightness = (0.5, 2, 0.5)
        self.size_train = (640, 1280)
        self.size_test = (1024, 2048)

        # image channel-wise mean to subtract, the order is BGR
        self.img_channel_mean = [103.939, 116.779, 123.68]

        # use teacher
        self.teacher = True

        self.test_path = './data/citypersons'

        # whether or not to do validation during training
        self.val = True
        self.val_frequency = 1