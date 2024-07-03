import glob
import os
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple

import numpy as np
import scipy
import torchio as tio
from skimage import morphology

import eval_metrics


class Dataset(metaclass=ABCMeta):
    data_path = None
    images_pair = None
    labels_pair = None
    keypoints_pair = None
    normalize_mode = False
    mode = None
    return_mode = None
    imgshape = None
    spacing = None
    classes = None

    def __init__(self, data_path: str, mode: int, normalize_mode: Optional[bool]) -> None:
        assert os.path.isdir(data_path), 'Please provide a valid data file. \n (not valid {})'.format(data_path)
        self.data_path = data_path
        assert (mode in ["train", "test", "val", "val2", "val4", "train4",
                         "test4"]), 'Please provide a valid return mode.'
        self.mode = mode
        self.normalize_mode = normalize_mode

    def __len__(self) -> int:
        """

        @return: number of samples/image pairs in dataset
        """
        return len(self.images_pair)

    @abstractmethod
    def __getitem__(self, item: int) -> Tuple[np.array, ...]:
        pass

    def get_initial_dice(self):  # TODO test
        assert self.return_mode > 2
        score = 0
        elements = 0
        for _, _, moving_labels, fixed_labels, _, _ in iter(self):
            score += eval_metrics.dice(moving_labels.squeeze(), fixed_labels.squeeze())
            elements += 1
        return score / elements

    def get_img_mask_hull(self, image):
        points = np.transpose(np.where(image))
        hull = scipy.spatial.ConvexHull(points)
        deln = scipy.spatial.Delaunay(points[hull.vertices])
        idx = np.stack(np.indices(image.shape), axis=-1)
        out_idx = np.nonzero(deln.find_simplex(idx) + 1)
        out_img = np.zeros(image.shape)
        out_img[out_idx] = 1
        return out_img


class Learn2RegLungCTDataset(Dataset):
    """
    Lung CT-CT dataset from from https://learn2reg.grand-challenge.org/Datasets/
    Train, test, val: 20,3,6

    original : origin/spacing/shape
    (0.0, 0.0, 0.0)
    (1.75, 1.25, 1.75)
    (192, 192, 208)
    """

    def __init__(self, data_path: str, mode: str, normalize_mode: Optional[bool] = True,
                 clip_mode: Optional[bool] = True, seg_mode: Optional[str] = "LungRibsLiver",
                 cropping: Optional[str] = None, dim_mode='3d', load_preprocessed=True) -> None:
        super(Learn2RegLungCTDataset, self).__init__(data_path, mode, normalize_mode)

        def _read_filenames():
            x_ls = list()
            y_ls = list()
            keypoints_x_ls = list()
            keypoints_y_ls = list()
            masks_x_ls = list()
            masks_y_ls = list()

            if self.mode == "train" or self.mode == "train4":
                # load train paths. 0=fixed, 1=moving
                for i in range(1, 21):
                    file_str = "LungCT_" + str(i).zfill(4)
                    x_ls.append(self.data_path + "/imagesTr/" + file_str + "_0001.nii.gz")
                    y_ls.append(self.data_path + "/imagesTr/" + file_str + "_0000.nii.gz")
                    keypoints_x_ls.append(self.data_path + "/keypointsTr/" + file_str + "_0001.csv")
                    keypoints_y_ls.append(self.data_path + "/keypointsTr/" + file_str + "_0000.csv")
                    if self.seg_mode == "LungRibsLiver":
                        masks_x_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0001.nii.gz")
                        masks_y_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0000.nii.gz")
                    elif self.seg_mode == "LungRibs":
                        masks_x_ls.append(self.data_path + "/masksLungRibs/" + file_str + "_0001.nii.gz")
                        masks_y_ls.append(self.data_path + "/masksLungRibs/" + file_str + "_0000.nii.gz")
                    elif self.seg_mode == "Lung":
                        masks_x_ls.append(self.data_path + "/masksTr/" + file_str + "_0001.nii.gz")
                        masks_y_ls.append(self.data_path + "/masksTr/" + file_str + "_0000.nii.gz")
            elif self.mode == "val" or self.mode == "val2":
                for i in range(21, 24):
                    file_str = "LungCT_" + str(i).zfill(4)
                    x_ls.append(self.data_path + "/imagesTs/" + file_str + "_0001.nii.gz")
                    y_ls.append(self.data_path + "/imagesTs/" + file_str + "_0000.nii.gz")
                    keypoints_x_ls.append(self.data_path + "/keypointsTs/" + file_str + "_0001.csv")
                    keypoints_y_ls.append(self.data_path + "/keypointsTs/" + file_str + "_0001.csv")
                    if self.seg_mode == "LungRibsLiver":
                        masks_x_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0001.nii.gz")
                        masks_y_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0000.nii.gz")
                    elif self.seg_mode == "LungRibs":
                        masks_x_ls.append(self.data_path + "/masksLungRibs/" + file_str + "_0001.nii.gz")
                        masks_y_ls.append(self.data_path + "/masksLungRibs/" + file_str + "_0000.nii.gz")
                    elif self.seg_mode == "Lung":
                        masks_x_ls.append(self.data_path + "/masksTs/" + file_str + "_0001.nii.gz")
                        masks_y_ls.append(self.data_path + "/masksTs/" + file_str + "_0000.nii.gz")
            else:
                for i in [24, 25, 28, 29]:  # 26 and 27 no liver ----> find solution in future!
                    file_str = "LungCT_" + str(i).zfill(4)
                    x_ls.append(self.data_path + "/imagesTs/" + file_str + "_0001.nii.gz")
                    y_ls.append(self.data_path + "/imagesTs/" + file_str + "_0000.nii.gz")
                    keypoints_x_ls.append(self.data_path + "/keypointsTs/" + file_str + "_0001.csv")
                    keypoints_y_ls.append(self.data_path + "/keypointsTs/" + file_str + "_0000.csv")
                    if self.seg_mode == "LungRibsLiver":
                        masks_x_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0001.nii.gz")
                        masks_y_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0000.nii.gz")
                    elif self.seg_mode == "LungRibs":
                        masks_x_ls.append(self.data_path + "/masksLungRibs/" + file_str + "_0001.nii.gz")
                        masks_y_ls.append(self.data_path + "/masksLungRibs/" + file_str + "_0000.nii.gz")
                    elif self.seg_mode == "Lung":
                        masks_x_ls.append(self.data_path + "/masksTs/" + file_str + "_0001.nii.gz")
                        masks_y_ls.append(self.data_path + "/masksTs/" + file_str + "_0000.nii.gz")

            return list(zip(x_ls, y_ls)), list(zip(masks_x_ls, masks_y_ls)), list(zip(keypoints_x_ls, keypoints_y_ls))

        assert (mode in ['train', 'val', 'test', 'val2', 'val4', 'train4']), 'Please provide a valid mode.'

        if dim_mode == '3d':
            self.ndim = 3
            self.spacing = (2, 2, 2)  # 1.75x1.25x1.75mm
            self.imgshape = (192, 128, 192)
        elif dim_mode == '2d':
            self.ndim = 2
            self.slice2d = 64
            self.spacing = (2, 2)  # 1.75x1.25x1.75mm
            self.imgshape = (192, 192)
        # self.imgshape = (182, 120, 168)
        self.data_path = data_path
        self.seg_mode = seg_mode
        self.load_preprocessed = load_preprocessed
        if self.load_preprocessed:
            self.data_path += "_preprocessed"
            print("LOADING PREPROCESSED LungCT DATA=TRUE")
        self.cropping = None
        if cropping:
            assert cropping in ['equal_borders', 'img_mask']
            self.cropping = cropping

        if seg_mode == "LungRibsLiver":
            self.classes = {0: "background", 1: "lung", 2: "bones", 3: "liver"}
        elif seg_mode == "LungRibs":
            self.classes = {0: "background", 1: "lung", 2: "bones"}
        elif seg_mode == "Lung":
            self.classes = {0: "background", 1: "lung"}

        if self.mode == "train" or self.mode == "val2":
            self.return_mode = 2
        elif self.mode == "val" or self.mode == "test":
            self.return_mode = 6
        elif self.mode == "train4" or self.mode == 'val4':
            self.return_mode = 4
        else:
            print("Wrong mode given")

        if self.return_mode == 6 and self.ndim == 2:
            self.return_mode = 4

        self.clip_mode = clip_mode

        self.images_pair, self.labels_pair, self.keypoints_pair = _read_filenames()

    def crop_img_mask(self, image_to_crop: np.ndarray, domain_image: Optional[np.ndarray]):
        img_mask = domain_image > 0
        cropped_img = morphology.convex_hull_image(img_mask).astype(float) * image_to_crop
        return cropped_img

    def crop_black_borders(self, image_to_crop: np.ndarray, domain_image: Optional[np.ndarray]):
        cropped_img = np.zeros_like(image_to_crop)
        if self.ndim == 2:
            cropped_img[np.ix_((domain_image > 1e-6).any(1), (domain_image > 1e-6).any(0))] = 1  # rows and cols
        elif self.ndim == 3:
            for dim in range(3):
                # Check if there is at least one element with value above 1e-6 along the current dimension
                mask = (domain_image > 1e-6).any(axis=dim, keepdims=True)

                if dim == 0:
                    mask = np.tile(mask, (image_to_crop.shape[0], 1, 1))
                    cropped_img[mask] += 1
                if dim == 1:
                    mask = np.tile(mask, (1, image_to_crop.shape[1], 1))
                    cropped_img[mask] += 1
                if dim == 2:
                    mask = np.tile(mask, (1, 1, image_to_crop.shape[2]))
                    cropped_img[mask] += 1
            cropped_img[cropped_img == 1] = 0
            cropped_img[cropped_img == 2] = 0
            cropped_img[cropped_img == 3] = 1
        cropped_img = image_to_crop * cropped_img
        return cropped_img

    def __getitem__(self, idx: int) -> Tuple[np.array, ...]:
        '''
        @param idx: Index of the item to return
        @return: numpy arrays
        '''
        if not self.load_preprocessed:
            x_file, y_file = self.images_pair[idx]

            if self.return_mode == 6:
                keypoints_x = np.genfromtxt(self.keypoints_pair[idx][0], delimiter=',')
                keypoints_y = np.genfromtxt(self.keypoints_pair[idx][1], delimiter=',')

                keypoints_x = keypoints_x[:, [2, 1, 0]]
                keypoints_y = keypoints_y[:, [2, 1, 0]]

                keypoints_x[:, 0] = keypoints_x[:, 0] * -1
                keypoints_y[:, 0] = keypoints_y[:, 0] * -1
                #
                keypoints_x[:, 0] = keypoints_x[:, 0] + 208
                keypoints_y[:, 0] = keypoints_y[:, 0] + 208
                #
                keypoints_x[:, 0] = keypoints_x[:, 0] * 1.75 / 2
                keypoints_x[:, 1] = keypoints_x[:, 1] * 1.25 / 2
                keypoints_x[:, 2] = keypoints_x[:, 2] * 1.75 / 2
                keypoints_y[:, 0] = keypoints_y[:, 0] * 1.75 / 2
                keypoints_y[:, 1] = keypoints_y[:, 1] * 1.25 / 2
                keypoints_y[:, 2] = keypoints_y[:, 2] * 1.75 / 2
                #
                keypoints_x[:, 0] = keypoints_x[:, 0] + 5
                keypoints_x[:, 1] = keypoints_x[:, 1] + 4
                keypoints_x[:, 2] = keypoints_x[:, 2] + 12
                keypoints_y[:, 0] = keypoints_y[:, 0] + 5
                keypoints_y[:, 1] = keypoints_y[:, 1] + 4
                keypoints_y[:, 2] = keypoints_y[:, 2] + 12

            if self.return_mode >= 4:
                labels_x_file, labels_y_file = self.labels_pair[idx]
                subject_dict = {"image_x": tio.ScalarImage(x_file), "labels_x": tio.LabelMap(labels_x_file),
                    "image_y": tio.ScalarImage(y_file), "labels_y": tio.LabelMap(labels_y_file)}
            else:
                subject_dict = {"image_x": tio.ScalarImage(x_file), "image_y": tio.ScalarImage(y_file), }

            subject = tio.Subject(subject_dict)

            resample_uniform = tio.Resample(2)
            subject = resample_uniform(subject)

            if self.clip_mode:
                clamp = tio.Clamp(out_min=-980, out_max=600)
                subject = clamp(subject)

            if self.normalize_mode:
                rescale_x = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), in_min_max=(
                    subject["image_x"].numpy().min(), subject["image_x"].numpy().max()))
                rescale_y = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), in_min_max=(
                    subject["image_y"].numpy().min(), subject["image_y"].numpy().max()))
                # rescale = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100))

                subject["image_x"] = rescale_x(subject["image_x"])
                # subject["image_x"] = rescale(subject["image_x"])
                subject["image_y"] = rescale_y(subject["image_y"])  # subject["image_y"] = rescale(subject["image_y"])

            crop_or_pad = tio.CropOrPad((192, 128, 192),
                                        padding_mode=0.0)  # before: (182, 120, 168), after: (192, 128, 192 ) - added (10,8, 24)
            subject = crop_or_pad(subject)

            if self.ndim == 2:
                subject["image_x"] = tio.ScalarImage(
                    tensor=subject["image_x"].numpy()[:, :, self.slice2d, :, np.newaxis])
                subject["image_y"] = tio.ScalarImage(
                    tensor=subject["image_y"].numpy()[:, :, self.slice2d, :, np.newaxis])

                if self.return_mode >= 4:
                    subject["labels_x"] = tio.ScalarImage(
                        tensor=subject["labels_x"].numpy()[:, :, self.slice2d, :, np.newaxis])
                    subject["labels_y"] = tio.ScalarImage(
                        tensor=subject["labels_y"].numpy()[:, :, self.slice2d, :, np.newaxis])

            x = np.flipud(subject["image_x"].numpy().transpose((0, 3, 2, 1)).squeeze())
            y = np.flipud(subject["image_y"].numpy().transpose((0, 3, 2, 1)).squeeze())

            if self.cropping == 'equal_borders':
                x = self.crop_black_borders(x, y)
                y = self.crop_black_borders(y, x)


            elif self.cropping == 'img_mask':
                x = self.crop_img_mask(x, y)

            if self.return_mode >= 4:
                labels_x = np.flipud(subject["labels_x"].numpy().transpose((0, 3, 2, 1)).squeeze())
                labels_y = np.flipud(subject["labels_y"].numpy().transpose((0, 3, 2, 1)).squeeze())

                if self.cropping == "equal_borders":
                    labels_x = self.crop_black_borders(labels_x, y)
                    labels_y = self.crop_black_borders(labels_y, x)
                if self.cropping == "img_mask":
                    labels_x = self.crop_img_mask(labels_x, y)
                    labels_y = self.crop_img_mask(labels_y, x)

            # from full images to cropped
            if self.return_mode == 6:
                return x.astype(float).squeeze(), y.astype(float).squeeze(), labels_x.astype(
                    float).squeeze(), labels_y.astype(float).squeeze(), keypoints_x, keypoints_y
            elif self.return_mode == 4:
                return x.astype(float).squeeze(), y.astype(float).squeeze(), labels_x.astype(
                    float).squeeze(), labels_y.astype(float).squeeze()
            else:
                return x.astype(float).squeeze(), y.astype(float).squeeze()
        else:
            x_file, y_file = self.images_pair[idx]

            if self.return_mode == 6:
                keypoints_x = np.genfromtxt(self.keypoints_pair[idx][0], delimiter=',')
                keypoints_y = np.genfromtxt(self.keypoints_pair[idx][1], delimiter=',')

            if self.return_mode >= 4:
                labels_x_file, labels_y_file = self.labels_pair[idx]
                subject_dict = {"image_x": tio.ScalarImage(x_file), "labels_x": tio.LabelMap(labels_x_file),
                    "image_y": tio.ScalarImage(y_file), "labels_y": tio.LabelMap(labels_y_file)}
            else:
                subject_dict = {"image_x": tio.ScalarImage(x_file), "image_y": tio.ScalarImage(y_file), }

            subject = tio.Subject(subject_dict)

            if self.ndim == 2:
                subject["image_x"] = tio.ScalarImage(
                    tensor=subject["image_x"].numpy()[:, :, self.slice2d, :, np.newaxis])
                subject["image_y"] = tio.ScalarImage(
                    tensor=subject["image_y"].numpy()[:, :, self.slice2d, :, np.newaxis])

                if self.return_mode >= 4:
                    subject["labels_x"] = tio.ScalarImage(
                        tensor=subject["labels_x"].numpy()[:, :, self.slice2d, :, np.newaxis])
                    subject["labels_y"] = tio.ScalarImage(
                        tensor=subject["labels_y"].numpy()[:, :, self.slice2d, :, np.newaxis])

            x = subject["image_x"].numpy()
            y = subject["image_y"].numpy()

            if self.return_mode >= 4:
                labels_x = subject["labels_x"].numpy()
                labels_y = subject["labels_y"].numpy()

            if self.return_mode == 6:
                return x.astype(float).squeeze(), y.astype(float).squeeze(), labels_x.astype(
                    float).squeeze(), labels_y.astype(float).squeeze(), keypoints_x, keypoints_y
            elif self.return_mode == 4:
                return x.astype(float).squeeze(), y.astype(float).squeeze(), labels_x.astype(
                    float).squeeze(), labels_y.astype(float).squeeze()
            else:
                return x.astype(float).squeeze(), y.astype(float).squeeze()


class NLST2023Dataset(Dataset):
    """
    NLST 2023 dataset from https://learn2reg.grand-challenge.org/Datasets/
    209 annotated images in total
    Train, test, val:  169, 30 (first 30 pairs), 10 (as in json file)
    """

    def __init__(self, data_path: str, mode: str, normalize_mode: Optional[bool] = True,
                 clip_mode: Optional[bool] = True, dim_mode='3d', cropping: Optional[str] = None, filter_air=True,
                 mini=False, load_preprocessed=True) -> None:
        super(NLST2023Dataset, self).__init__(data_path, mode, normalize_mode)

        def _read_filenames(mini):
            x_ls = list()
            y_ls = list()
            keypoints_x_ls = list()
            keypoints_y_ls = list()
            masks_x_ls = list()
            masks_y_ls = list()

            if self.mode == "train" or self.mode == "train4":
                # load train paths. 0=fixed, 1=moving
                for i in range(30, 101):
                    file_str = "NLST_" + str(i).zfill(4)
                    y_ls.append(self.data_path + "/imagesTr/" + file_str + "_0000.nii.gz")
                    x_ls.append(self.data_path + "/imagesTr/" + file_str + "_0001.nii.gz")
                    if self.load_preprocessed:
                        masks_y_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0000.nii.gz")
                        masks_x_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0001.nii.gz")
                    else:
                        masks_y_ls.append(self.data_path[:-5] + "/masksLungRibsLiver/" + file_str + "_0000.nii.gz")
                        masks_x_ls.append(self.data_path[:-5] + "/masksLungRibsLiver/" + file_str + "_0001.nii.gz")

                for i in range(200, 300):
                    file_str = "NLST_" + str(i).zfill(4)
                    y_ls.append(self.data_path + "/imagesTr/" + file_str + "_0000.nii.gz")
                    x_ls.append(self.data_path + "/imagesTr/" + file_str + "_0001.nii.gz")
                    if self.load_preprocessed:
                        masks_y_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0000.nii.gz")
                        masks_x_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0001.nii.gz")
                    else:
                        masks_y_ls.append(self.data_path[:-5] + "/masksLungRibsLiver/" + file_str + "_0000.nii.gz")
                        masks_x_ls.append(self.data_path[:-5] + "/masksLungRibsLiver/" + file_str + "_0001.nii.gz")

            elif self.mode == "val" or self.mode == "val2" or self.mode == "val4":
                for i in range(101, 111):
                    file_str = "NLST_" + str(i).zfill(4)
                    y_ls.append(self.data_path + "/imagesTr/" + file_str + "_0000.nii.gz")
                    x_ls.append(self.data_path + "/imagesTr/" + file_str + "_0001.nii.gz")
                    keypoints_y_ls.append(self.data_path + "/keypointsTr/" + file_str + "_0000.csv")
                    keypoints_x_ls.append(self.data_path + "/keypointsTr/" + file_str + "_0001.csv")

                    if self.load_preprocessed:
                        masks_y_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0000.nii.gz")
                        masks_x_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0001.nii.gz")
                    else:
                        masks_y_ls.append(self.data_path[:-5] + "/masksLungRibsLiver/" + file_str + "_0000.nii.gz")
                        masks_x_ls.append(self.data_path[:-5] + "/masksLungRibsLiver/" + file_str + "_0001.nii.gz")
            else:
                for i in range(1, 31):
                    file_str = "NLST_" + str(i).zfill(4)
                    y_ls.append(self.data_path + "/imagesTr/" + file_str + "_0000.nii.gz")
                    x_ls.append(self.data_path + "/imagesTr/" + file_str + "_0001.nii.gz")
                    keypoints_y_ls.append(self.data_path + "/keypointsTr/" + file_str + "_0000.csv")
                    keypoints_x_ls.append(self.data_path + "/keypointsTr/" + file_str + "_0001.csv")
                    # masks_y_ls.append(self.data_path + "/masksTr/" + file_str + "_0000.nii.gz")
                    if self.load_preprocessed:
                        masks_y_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0000.nii.gz")
                        masks_x_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0001.nii.gz")
                    else:
                        masks_y_ls.append(self.data_path[:-5] + "/masksLungRibsLiver/" + file_str + "_0000.nii.gz")
                        masks_x_ls.append(self.data_path[:-5] + "/masksLungRibsLiver/" + file_str + "_0001.nii.gz")

            if mini:
                if self.mode == "train" or self.mode == "train4":
                    y_ls = y_ls[:20]
                    x_ls = x_ls[:20]
                    masks_x_ls = masks_x_ls[:20]
                    masks_y_ls = masks_y_ls[:20]
                elif self.mode == "val" or self.mode == "val2" or self.mode == "val4":
                    y_ls = y_ls[:3]
                    x_ls = x_ls[:3]
                    masks_x_ls = masks_x_ls[:3]
                    masks_y_ls = masks_y_ls[:3]
                    keypoints_x_ls = keypoints_x_ls[:3]
                    keypoints_y_ls = keypoints_y_ls[:3]
                else:
                    y_ls = y_ls[:6]
                    x_ls = x_ls[:6]
                    masks_x_ls = masks_x_ls[:6]
                    masks_y_ls = masks_y_ls[:6]
                    keypoints_x_ls = keypoints_x_ls[:6]
                    keypoints_y_ls = keypoints_y_ls[:6]

            return list(zip(x_ls, y_ls)), list(zip(masks_x_ls, masks_y_ls)), list(zip(keypoints_x_ls, keypoints_y_ls))

        assert (mode in ['train', 'train4', 'val', 'test', 'test4', 'val4', 'val2']), 'Please provide a valid mode.'

        if dim_mode == '3d':
            self.ndim = 3
            self.spacing = (1.5, 1.5, 1.5)
            self.imgshape = (224, 192, 224)
        elif dim_mode == '2d':
            self.ndim = 2
            self.slice2d = 96
            self.spacing = (1.5, 1.5)
            self.imgshape = (224, 224)  # slice middle dim
        else:
            print("not implemented")

        self.classes = {0: "background", 1: "lung", 2: "bones", 3: "liver"}

        self.filter_air = filter_air
        self.cropping = None
        if cropping:
            assert cropping in ['equal_borders', 'img_mask']
            self.cropping = cropping

        self.mini = mini
        self.load_preprocessed = load_preprocessed
        if self.load_preprocessed:
            self.data_path += "_preprocessed"
            print("LOADING PREPROCESSED NLST DATA=TRUE")
        if dim_mode == "2d":
            self.data_path += "_2d"

        if self.mode == "train" or self.mode == "val2":
            self.return_mode = 2
        elif self.mode == "train4" or self.mode == "val4" or self.mode == "test4":
            self.return_mode = 4
        elif self.mode == "val" or self.mode == "test":
            self.return_mode = 6
        else:
            print("Wrong mode given")

        self.clip_mode = clip_mode
        self.images_pair, self.labels_pair, self.keypoints_pair = _read_filenames(self.mini)

    def crop_img_mask(self, image_to_crop: np.ndarray, domain_image: Optional[np.ndarray]):
        cropped_img = np.zeros_like(image_to_crop)
        if self.ndim == 2:
            img_mask = domain_image > 1e-1  # tested fairly well, not fully
            cropped_img = morphology.convex_hull_image(img_mask).astype(float)
        elif self.ndim == 3:
            print('not implemented')
        return cropped_img * image_to_crop

    def crop_black_borders(self, image_to_crop: np.ndarray, domain_image: Optional[np.ndarray]):
        cropped_img = np.zeros_like(image_to_crop)
        if self.ndim == 2:
            cropped_img[np.ix_((domain_image > 1e-6).any(1), (domain_image > 1e-6).any(0))] = 1  # rows and cols
        elif self.ndim == 3:
            for dim in range(3):
                # Check if there is at least one element with value above 1e-6 along the current dimension
                mask = (domain_image > 1e-6).any(axis=dim, keepdims=True)

                if dim == 0:
                    mask = np.tile(mask, (image_to_crop.shape[0], 1, 1))
                    cropped_img[mask] += 1
                if dim == 1:
                    mask = np.tile(mask, (1, image_to_crop.shape[1], 1))
                    cropped_img[mask] += 1
                if dim == 2:
                    mask = np.tile(mask, (1, 1, image_to_crop.shape[2]))
                    cropped_img[mask] += 1
            cropped_img[cropped_img == 1] = 0
            cropped_img[cropped_img == 2] = 0
            cropped_img[cropped_img == 3] = 1
        cropped_img = image_to_crop * cropped_img
        return cropped_img

    def __getitem__(self, idx: int) -> Tuple[np.array, ...]:
        '''
        @param idx: Index of the item to return
        @return: numpy arrays
        '''

        if self.load_preprocessed is False:
            x_file, y_file = self.images_pair[idx]
            if self.return_mode == 6:
                keypoints_x = np.genfromtxt(self.keypoints_pair[idx][0], delimiter=',')
                keypoints_y = np.genfromtxt(self.keypoints_pair[idx][1], delimiter=',')
                keypoints_x = keypoints_x[:, [2, 1, 0]]
                keypoints_y = keypoints_y[:, [2, 1, 0]]

                keypoints_x[:, 0] = keypoints_x[:, 0] * -1
                keypoints_y[:, 0] = keypoints_y[:, 0] * -1

                keypoints_x[:, 0] = keypoints_x[:, 0] + 224
                keypoints_y[:, 0] = keypoints_y[:, 0] + 224

            if self.return_mode >= 4:
                labels_x_file, labels_y_file = self.labels_pair[idx]
                subject_dict = {"image_x": tio.ScalarImage(x_file), "labels_x": tio.LabelMap(labels_x_file),
                    "image_y": tio.ScalarImage(y_file), "labels_y": tio.LabelMap(labels_y_file)}
            else:
                subject_dict = {"image_x": tio.ScalarImage(x_file), "image_y": tio.ScalarImage(y_file), }

            subject = tio.Subject(subject_dict)

            if self.ndim == 2:

                subject["image_x"] = tio.ScalarImage(
                    tensor=subject["image_x"].numpy()[:, :, self.slice2d, :, np.newaxis])
                subject["image_y"] = tio.ScalarImage(
                    tensor=subject["image_y"].numpy()[:, :, self.slice2d, :, np.newaxis])

                if self.return_mode >= 4:
                    subject["labels_x"] = tio.ScalarImage(
                        tensor=subject["labels_x"].numpy()[:, :, self.slice2d, :, np.newaxis])
                    subject["labels_y"] = tio.ScalarImage(
                        tensor=subject["labels_y"].numpy()[:, :, self.slice2d, :, np.newaxis])

            if self.filter_air:
                # mask of body
                y_mask = (np.flipud(subject["image_y"].numpy().transpose((0, 3, 2, 1)).squeeze()) > -20)
                y_mask_hull = self.get_img_mask_hull(y_mask)

                x_mask = (np.flipud(subject["image_x"].numpy().transpose((0, 3, 2, 1)).squeeze()) > -20)
                x_mask_hull = self.get_img_mask_hull(x_mask)

            if self.clip_mode:
                clamp = tio.Clamp(out_min=-980, out_max=1518)  # tony mok github comment -1100, 1518
                subject = clamp(subject)

            if self.normalize_mode:
                rescale_x = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), in_min_max=(
                    subject["image_x"].numpy().min(), subject["image_x"].numpy().max()))
                rescale_y = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), in_min_max=(
                    subject["image_y"].numpy().min(), subject["image_y"].numpy().max()))
                subject["image_x"] = rescale_x(subject["image_x"])
                subject["image_y"] = rescale_y(subject["image_y"])

            x = np.flipud(subject["image_x"].numpy().transpose((0, 3, 2, 1)).squeeze())
            y = np.flipud(subject["image_y"].numpy().transpose((0, 3, 2, 1)).squeeze())

            if self.filter_air:
                x = x * x_mask_hull
                y = y * y_mask_hull

            if self.cropping == 'equal_borders':
                x = self.crop_black_borders(x, y)
                y = self.crop_black_borders(y, x)
            elif self.cropping == 'img_mask':
                x = self.crop_img_mask(x, y)
                y = self.crop_img_mask(y, x)

            if self.return_mode == 6:
                labels_x = np.flipud(subject["labels_x"].numpy().transpose((0, 3, 2, 1)).squeeze())
                labels_y = np.flipud(subject["labels_y"].numpy().transpose((0, 3, 2, 1)).squeeze())
                if self.ndim == 3:
                    return x.astype(float), y.astype(float), labels_x.astype(np.int16), labels_y.astype(
                        np.int16), keypoints_x, keypoints_y
                else:
                    return x.astype(float).squeeze()[:, self.slice2d, :], y.astype(float).squeeze()[:, self.slice2d,
                                                                          :], labels_x.astype(np.int16).squeeze()[:,
                                                                              self.slice2d, :], labels_y.astype(
                        np.int16).squeeze()[:, self.slice2d, :], keypoints_x, keypoints_y
            elif self.return_mode == 4:
                labels_x = np.flipud(subject["labels_x"].numpy().transpose((0, 3, 2, 1)).squeeze())
                labels_y = np.flipud(subject["labels_y"].numpy().transpose((0, 3, 2, 1)).squeeze())
                return x.astype(float), y.astype(float), labels_x.astype(np.int16), labels_y.astype(np.int16)
            else:
                return x.astype(float), y.astype(float)
        else:
            x_file, y_file = self.images_pair[idx]
            if self.return_mode == 6:
                keypoints_x = np.genfromtxt(self.keypoints_pair[idx][0], delimiter=',')
                keypoints_y = np.genfromtxt(self.keypoints_pair[idx][1], delimiter=',')

            if self.return_mode >= 4:
                labels_x_file, labels_y_file = self.labels_pair[idx]
                subject_dict = {"image_x": tio.ScalarImage(x_file), "labels_x": tio.LabelMap(labels_x_file),
                    "image_y": tio.ScalarImage(y_file), "labels_y": tio.LabelMap(labels_y_file)}
            else:
                subject_dict = {"image_x": tio.ScalarImage(x_file), "image_y": tio.ScalarImage(y_file), }

            subject = tio.Subject(subject_dict)

            x = subject["image_x"].numpy().squeeze()
            y = subject["image_y"].numpy().squeeze()

            if self.return_mode == 6:
                labels_x = subject["labels_x"].numpy().squeeze()
                labels_y = subject["labels_y"].numpy().squeeze()
                return x.astype(float), y.astype(float), labels_x.astype(np.int16), labels_y.astype(
                    np.int16), keypoints_x, keypoints_y
            elif self.return_mode == 4:
                labels_x = subject["labels_x"].numpy().squeeze()
                labels_y = subject["labels_y"].numpy().squeeze()
                return x.astype(float), y.astype(float), labels_x.astype(np.int16), labels_y.astype(np.int16)
            else:
                return x.astype(float), y.astype(float)


class ACDCDataset(Dataset):
    """
    source: https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb
    Number of subjects: 150
    modality: MR
    anatomy: cardiac
    m,f: fixed=ed/01, moving=es/1x

    split the 50 test samples in 20 for val and 30 for testing
    """

    def __init__(self, data_path: str, mode: str, normalize_mode: Optional[bool] = True,
                 roi_only: Optional[bool] = True, dim_mode: Optional[str] = '3d') -> None:

        super(ACDCDataset, self).__init__(data_path, mode, normalize_mode)

        def _read_filenames():
            x_ls = list()
            y_ls = list()
            masks_x_ls = list()
            masks_y_ls = list()
            group_ls = list()

            if self.mode == "train" or self.mode == "train4":
                # load train paths. 01=fixed, 1x=moving
                for i in range(1, 101):
                    file_str = "patient" + str(i).zfill(3)
                    file_str_full = self.data_path + '/training/' + file_str + "/" + file_str + '_frame*_gt.nii.gz'
                    ids = sorted(glob.glob(file_str_full))
                    m_id = ids[1][-12:-10]
                    f_id = ids[0][-12:-10]
                    # print(m_id, f_id)
                    y_ls.append(self.data_path + "/training/" + file_str + "/" + file_str + "_frame" + f_id + ".nii.gz")
                    masks_y_ls.append(
                        self.data_path + "/training/" + file_str + "/" + file_str + "_frame" + f_id + "_gt.nii.gz")
                    x_ls.append(self.data_path + "/training/" + file_str + "/" + file_str + "_frame" + m_id + ".nii.gz")
                    masks_x_ls.append(
                        self.data_path + "/training/" + file_str + "/" + file_str + "_frame" + m_id + "_gt.nii.gz")
                    with open(self.data_path + "/training/" + file_str + "/Info.cfg", 'r') as file:
                        for i, line in enumerate(file):
                            if i == 2:
                                group = line.split(':')[1].strip()
                                break

                    group_ls.append(self.group_map[group])

            elif self.mode == "val" or self.mode == "val2" or self.mode == "val4":
                for i in range(101, 121):
                    file_str = "patient" + str(i).zfill(3)
                    file_str_full = self.data_path + "/testing/" + file_str + "/" + file_str + "_frame*_gt.nii.gz"
                    ids = sorted(glob.glob(file_str_full))
                    m_id = ids[1][-12:-10]
                    f_id = ids[0][-12:-10]
                    y_ls.append(self.data_path + "/testing/" + file_str + "/" + file_str + "_frame" + f_id + ".nii.gz")
                    masks_y_ls.append(
                        self.data_path + "/testing/" + file_str + "/" + file_str + "_frame" + f_id + "_gt.nii.gz")
                    x_ls.append(self.data_path + "/testing/" + file_str + "/" + file_str + "_frame" + m_id + ".nii.gz")
                    masks_x_ls.append(
                        self.data_path + "/testing/" + file_str + "/" + file_str + "_frame" + m_id + "_gt.nii.gz")
                    with open(self.data_path + "/testing/" + file_str + "/Info.cfg", 'r') as file:
                        for i, line in enumerate(file):
                            if i == 2:
                                group = line.split(':')[1].strip()
                                break

                    group_ls.append(self.group_map[group])
            else:
                for i in range(121, 151):
                    file_str = "patient" + str(i).zfill(3)
                    file_str_full = self.data_path + "/testing/" + file_str + "/" + file_str + "_frame*_gt.nii.gz"
                    ids = sorted(glob.glob(file_str_full))
                    m_id = ids[1][-12:-10]
                    f_id = ids[0][-12:-10]
                    y_ls.append(self.data_path + "/testing/" + file_str + "/" + file_str + "_frame" + f_id + ".nii.gz")
                    masks_y_ls.append(
                        self.data_path + "/testing/" + file_str + "/" + file_str + "_frame" + f_id + "_gt.nii.gz")
                    x_ls.append(self.data_path + "/testing/" + file_str + "/" + file_str + "_frame" + m_id + ".nii.gz")
                    masks_x_ls.append(
                        self.data_path + "/testing/" + file_str + "/" + file_str + "_frame" + m_id + "_gt.nii.gz")
                    with open(self.data_path + "/testing/" + file_str + "/Info.cfg", 'r') as file:
                        for i, line in enumerate(file):
                            if i == 2:
                                group = line.split(':')[1].strip()
                                break

                    group_ls.append(self.group_map[group])

            return list(zip(x_ls, y_ls)), list(zip(masks_x_ls, masks_y_ls)), group_ls

        assert (mode in ['train', 'train3', 'train4', 'val', 'test', 'val2', 'val3', 'val4',
                         'test3']), 'Please provide a valid mode.'
        if roi_only:
            self.imgshape = (128, 128, 128)
        self.spacing = (1.8, 1.8, 1.8)  # as in Qin et al. 2023 MIA
        if dim_mode == '2d-random' or dim_mode == '2d-middle' or dim_mode == '2d-basal' and roi_only:
            self.spacing = (1.8, 1.8)
            self.imgshape = (128, 128)
        assert dim_mode in ['2d-random', '2d-middle', '2d-basal', '3d']
        self.dim_mode = dim_mode
        self.normalize_mode = normalize_mode
        self.classes = {  # TODO check!
            0: "background", 1: "RV",  # right ventricle
            2: "LV-Myo",  # epicardium
            3: "LV-BP"  # endocardium
        }

        self.group_map = {"NOR": 0, "MINF": 1, "DCM": 2, "HCM": 3, "RV": 4}

        self.roi_only = roi_only

        if "test" in self.mode or "val" in self.mode:
            if self.dim_mode == "2d-random":
                print("Manually set self.dim_mode for test/validation from random to middle!!")
                self.dim_mode = "2d-middle"

        if self.mode == "train" or self.mode == "val2":
            self.return_mode = 2
        elif self.mode == "train4" or self.mode == "val4" or self.mode == "val" or self.mode == "test":
            self.return_mode = 4
        elif self.mode == "train3" or self.mode == "val3" or self.mode == "test3":  # classification task
            self.return_mode = 3
        else:
            print("Wrong mode given")

        self.images_pair, self.labels_pair, self.groups = _read_filenames()

    def __getitem__(self, idx: int) -> Tuple[np.array, ...]:
        '''
        @param idx: Index of the item to return
        @return: numpy arrays

        2d-random: returns random 2d slice of size (HxW)
        '''

        x_file, y_file = self.images_pair[idx]
        group = self.groups[idx]

        labels_x_file, labels_y_file = self.labels_pair[idx]
        subject_dict = {"image_x": tio.ScalarImage(x_file), "labels_x": tio.LabelMap(labels_x_file),
            "image_y": tio.ScalarImage(y_file), "labels_y": tio.LabelMap(labels_y_file), "group": group}

        subject = tio.Subject(subject_dict)

        resample_uniform = tio.Resample(1.8)
        subject = resample_uniform(subject)

        if self.roi_only is True:
            crop_roi = tio.CropOrPad((128, 128, 128), mask_name="labels_y")
            subject = crop_roi(subject)

        if self.dim_mode == '2d-random':
            # not all slices have all three labels, thus find random slice with all labels present

            while True:
                slice = np.random.randint(2, subject["image_x"].numpy().shape[-1] - 2)
                labels_x_tmp = subject["labels_x"].numpy()[..., slice, np.newaxis]
                labels_y_tmp = subject["labels_y"].numpy()[..., slice, np.newaxis]
                if len(np.unique(labels_x_tmp)) == 4 and len(np.unique(labels_y_tmp)) == 4:
                    break

                # print("not all labels present in slice. sample again.")
            subject["image_x"] = tio.ScalarImage(tensor=subject["image_x"].numpy()[..., slice, np.newaxis])
            subject["image_y"] = tio.ScalarImage(tensor=subject["image_y"].numpy()[..., slice, np.newaxis])
            subject["labels_x"] = tio.ScalarImage(tensor=labels_x_tmp)
            subject["labels_y"] = tio.ScalarImage(tensor=labels_y_tmp)

        if self.dim_mode == '2d-middle':
            slice = subject["image_x"].numpy().shape[-1] // 2
            labels_x_tmp = subject["labels_x"].numpy()[..., slice, np.newaxis]
            labels_y_tmp = subject["labels_y"].numpy()[..., slice, np.newaxis]
            if len(np.unique(labels_x_tmp)) < 4 or len(np.unique(labels_y_tmp)) < 4:
                print("not all labels present in slice.")

            subject["image_x"] = tio.ScalarImage(tensor=subject["image_x"].numpy()[..., slice, np.newaxis])
            subject["image_y"] = tio.ScalarImage(tensor=subject["image_y"].numpy()[..., slice, np.newaxis])
            subject["labels_x"] = tio.ScalarImage(tensor=labels_x_tmp)
            subject["labels_y"] = tio.ScalarImage(tensor=labels_y_tmp)

        if self.normalize_mode is True:
            rescale_x = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), in_min_max=(
                subject["image_x"].numpy().min(), subject["image_x"].numpy().max()))
            rescale_y = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), in_min_max=(
                subject["image_y"].numpy().min(), subject["image_y"].numpy().max()))

            subject["image_x"] = rescale_x(subject["image_x"])
            subject["image_y"] = rescale_y(subject["image_y"])

        x = subject["image_x"].numpy().astype(float).squeeze()
        y = subject["image_y"].numpy().astype(float).squeeze()
        labels_x = subject["labels_x"].numpy().astype(float).squeeze()
        labels_y = subject["labels_y"].numpy().astype(float).squeeze()

        if self.return_mode == 3:
            return x, y, group
        elif self.return_mode == 4:
            return x, y, labels_x, labels_y
        else:
            return x, y
