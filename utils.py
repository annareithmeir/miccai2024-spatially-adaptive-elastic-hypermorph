from typing import Optional
from collections.abc import Generator
import utils
import datasets
import numpy as np
import scipy
from scipy.ndimage import map_coordinates


def flood_fill_hull(image):
    """
    for image mask
    """
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(image.shape), axis=-1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img


def scan_to_scan_generator(dataset: datasets.Dataset, batch_size: Optional[int] = 1) -> Generator:
    """
    The basis generator for the desired dataset from the original voxelmorph code
    invols = [x, y]
    outvols = [y, zeros]

    :param dataset: dataset
    :param bidir:
    :param batch_size: batch size
    :param no_warp: no initial warp given
    :return: Generator with data of form (invols[m,f], outvols[m,f], segvols[m,f], kps[m,f])
    """

    return_mode = dataset.return_mode

    if batch_size>1:
        while True:
            indices = np.random.randint(len(dataset), size=batch_size) # e.g. for training, random batches

            if return_mode == 2:
                imgs_x=list()
                imgs_y=list()
                for idx in indices:
                    x, y = dataset.__getitem__(idx)
                    x = x[np.newaxis, ..., np.newaxis]
                    y = y[np.newaxis, ..., np.newaxis]
                    imgs_x.append(x)
                    imgs_y.append(y)
                x=np.concatenate(imgs_x, axis=0)
                y=np.concatenate(imgs_y, axis=0)
            if return_mode == 4:
                imgs_x = list()
                imgs_y = list()
                segs_x = list()
                segs_y = list()
                for idx in indices:
                    x, y, seg_x, seg_y = dataset.__getitem__(idx)
                    x = x[np.newaxis, ..., np.newaxis]
                    y = y[np.newaxis, ..., np.newaxis]
                    seg_x = seg_x[np.newaxis, ..., np.newaxis]
                    seg_y = seg_y[np.newaxis, ..., np.newaxis]
                    imgs_x.append(x)
                    imgs_y.append(y)
                    segs_x.append(seg_x)
                    segs_y.append(seg_y)
                x = np.concatenate(imgs_x, axis=0)
                y = np.concatenate(imgs_y, axis=0)
                seg_x = np.concatenate(segs_x, axis=0)
                seg_y = np.concatenate(segs_y, axis=0)
            if return_mode == 6:
                imgs_x = list()
                imgs_y = list()
                segs_x = list()
                segs_y = list()
                kps_x = list()
                kps_y = list()
                for idx in indices:
                    x, y, seg_x, seg_y, kp_x, kp_y = dataset.__getitem__(idx)
                    x = x[np.newaxis, ..., np.newaxis]
                    y = y[np.newaxis, ..., np.newaxis]
                    seg_x = seg_x[np.newaxis, ..., np.newaxis]
                    seg_y = seg_y[np.newaxis, ..., np.newaxis]
                    imgs_x.append(x)
                    imgs_y.append(y)
                    segs_x.append(seg_x)
                    segs_y.append(seg_y)
                    kps_x.append(kp_x)
                    kps_y.append(kp_y)
                x = np.concatenate(imgs_x, axis=0)
                y = np.concatenate(imgs_y, axis=0)
                seg_x = np.concatenate(segs_x, axis=0)
                seg_y = np.concatenate(segs_y, axis=0)
                kp_x=kps_x
                kp_y=kps_y

            shape = x.shape[1:-1]
            zeros = np.zeros((batch_size, *shape, len(shape)))

            invols = [x, y]
            outvols = [y]
            outvols.append(zeros)

            idx += 1
            if idx == len(dataset):
                idx = 0

            if return_mode > 4:
                kps = [kp_x, kp_y]
                segvols = [seg_x, seg_y]
                yield (invols, outvols, segvols, kps)
            elif return_mode > 2:
                segvols = [seg_x, seg_y]
                yield (invols, outvols, segvols)
            else:
                yield (invols, outvols)
    else:
        idx=0
        while True:
            if return_mode == 2:
                x, y = dataset.__getitem__(idx)
                x = x[np.newaxis, ..., np.newaxis]
                y = y[np.newaxis, ..., np.newaxis]
            if return_mode == 4:
                x, y, seg_x, seg_y = dataset.__getitem__(idx)
                x = x[np.newaxis, ..., np.newaxis]
                y = y[np.newaxis, ..., np.newaxis]
                seg_x = seg_x[np.newaxis, ..., np.newaxis]
                seg_y = seg_y[np.newaxis, ..., np.newaxis]

            if return_mode == 6:
                x, y, seg_x, seg_y, kp_x, kp_y = dataset.__getitem__(idx)
                x = x[np.newaxis, ..., np.newaxis]
                y = y[np.newaxis, ..., np.newaxis]
                seg_x = seg_x[np.newaxis, ..., np.newaxis]
                seg_y = seg_y[np.newaxis, ..., np.newaxis]

            shape = x.shape[1:-1]
            zeros = np.zeros((batch_size, *shape, len(shape)))

            invols = [x, y]
            outvols = [y]
            outvols.append(zeros)

            idx += 1
            if idx == len(dataset):
                idx = 0

            if return_mode > 4:
                kps = [kp_x, kp_y]
                segvols = [seg_x, seg_y]
                yield (invols, outvols, segvols, kps)
            elif return_mode > 2:
                segvols = [seg_x, seg_y]
                yield (invols, outvols, segvols)
            else:
                yield (invols, outvols)


def scan_to_scan_generator_sa(dataset: datasets.Dataset, bidir: Optional[bool] = False, batch_size: Optional[int] = 1,
                           no_warp: Optional[bool] = False, mode:Optional[str]="train4") -> Generator:
    """
    The basis generator for the desired dataset from the original voxelmorph code.
    invols = [x, y]
    outvols = [y, seg_x.astype(np.int16)]

    :param dataset: dataset
    :param bidir:
    :param batch_size: batch size
    :param no_warp: no initial warp given
    :return: Generator with data of form (invols[m,f], outvols[m,f], segvols[m,f], kps[m,f]) for val/test and (invols[m,f, seg_m], outvols[m,f])
    """

    return_mode = dataset.return_mode
    assert return_mode>2, "in scan2scan_sa: return mode must be greater than two."

    idx=0
    while True:
        if "test" in dataset.mode:
            indices=[idx]
            idx+=1
        else:
            indices = np.random.randint(len(dataset), size=batch_size) # e.g. for training, radnom batches

        if return_mode == 4:
            imgs_x = list()
            imgs_y = list()
            segs_x = list()
            segs_y = list()
            for idx in indices:
                x, y, seg_x, seg_y = dataset.__getitem__(idx)
                x = x[np.newaxis, ..., np.newaxis]
                y = y[np.newaxis, ..., np.newaxis]
                seg_x = seg_x[np.newaxis, ..., np.newaxis]
                seg_y = seg_y[np.newaxis, ..., np.newaxis]
                imgs_x.append(x)
                imgs_y.append(y)
                segs_x.append(seg_x)
                segs_y.append(seg_y)
            x = np.concatenate(imgs_x, axis=0)
            y = np.concatenate(imgs_y, axis=0)
            seg_x = np.concatenate(segs_x, axis=0)
            seg_y = np.concatenate(segs_y, axis=0)
        if return_mode == 6:
            imgs_x = list()
            imgs_y = list()
            segs_x = list()
            segs_y = list()
            kps_x = list()
            kps_y = list()
            for idx in indices:
                x, y, seg_x, seg_y, kp_x, kp_y = dataset.__getitem__(idx)
                x = x[np.newaxis, ..., np.newaxis]
                y = y[np.newaxis, ..., np.newaxis]
                seg_x = seg_x[np.newaxis, ..., np.newaxis]
                seg_y = seg_y[np.newaxis, ..., np.newaxis]
                imgs_x.append(x)
                imgs_y.append(y)
                segs_x.append(seg_x)
                segs_y.append(seg_y)
                kps_x.append(kp_x)
                kps_y.append(kp_y)
            x = np.concatenate(imgs_x, axis=0)
            y = np.concatenate(imgs_y, axis=0)
            seg_x = np.concatenate(segs_x, axis=0)
            seg_y = np.concatenate(segs_y, axis=0)


            if batch_size>1:
                kp_x = kps_x
                kp_y = kps_y

        invols = [x, y]
        outvols = [y, seg_y.astype(np.int16)] # put weight_tensor as true prediction s.t. it can be accessed in the loss grad function

        idx += 1
        if idx == len(dataset):
            idx = 0

        if return_mode > 4:
            kps = [kp_x, kp_y]
            segvols = [seg_x, seg_y]
            yield (invols, outvols, segvols, kps)
        elif return_mode > 2:
            segvols = [seg_x, seg_y]
            yield (invols, outvols, segvols)
        else:
            yield (invols, outvols)


def random_hyperparam() -> float:
    """
    Randomly samples a hyperparameter. oversample_rate is the fraction of samples from the range boundaries, i.e. 0 or 1
    :return: random value in [0,1]
    """
    oversample_rate = 0.2
    if np.random.rand() < oversample_rate:
        return np.random.choice([0, 1])
    else:
        return np.random.rand()


def random_hyperparam_in_range(sampling_range: Optional[list] = [0, 1]):
    """
        Randomly samples a hyperparameter from a predefined range [min,max]. oversample_rate is the fraction of samples from the range boundaries, i.e. min or max
        :return: random value in [min, max]
    """
    oversample_rate = 0.2
    if np.random.rand() < oversample_rate:
        return np.random.choice([sampling_range[0], sampling_range[1]])
    else:
        return np.random.uniform(low=sampling_range[0], high=sampling_range[1])


def hyp_generator(batch_size: int, base_generator: Generator, ranges: Optional[list] = None) -> Generator:
    """
    Generator for training of Hypermorph.
    inputs = [x, y, hyp]
    outputs = [y, y]
    # utputs = [[y, img_mask], y]

    :param nb_classes: nb_classes times sampling performed, if spatially adaptive is true. else once sampled and same weight is returned x times
    :param spatially_adaptive: creates reg weight array with nb_classes weights sampled, if true. if false only once sampled
    :param batch_size: batch size
    :param base_generator: the base generator for the desired dataset
    :param ranges: if desired, range for hyperparameter [min,max]. If not given then hyperparameter drawn from [0,1]
    :return: Generator with data of form (invols[m,f, hyperparam], outvols[m,f], segvols[m,f], kps[m,f]) OR
                    spattially-adaptive: (invols[m,f, hyp_tensor], outvols[m,f], segvols[m,f], kps[m,f])
    """
    while True:
        inputs, outputs = next(base_generator)
        if ranges is None:
            hyp = np.expand_dims([random_hyperparam() for _ in range(batch_size)], -1)
        else:
            hyp = np.expand_dims([random_hyperparam_in_range(ranges) for _ in range(batch_size)], -1)

        inputs = (*inputs, hyp)

        yield inputs, outputs


def hyp_generator_sa_vector(batch_size: int, base_generator: Generator, nb_classes: int, ranges: Optional[list] = None) -> Generator:
    """
    Generator for training of Hypermorph, spatially adaptive with vector-only input.
    inputs = [x, y, hyp]
    outputs = [[y, hyp_tensor], [y, hyp_tensor]]
    # outputs = [[y, hyp_tensor, img_mask], hyp_tensor]

    :param nb_classes: nb_classes times sampling performed, if spatially adaptive is true. else once sampled and same weight is returned x times
    :param spatially_adaptive: creates reg weight array with nb_classes weights sampled, if true. if false only once sampled
    :param batch_size: batch size
    :param base_generator: the base generator for the desired dataset
    :param ranges: if desired, range for hyperparameter [min,max]. If not given then hyperparameter drawn from [0,1]
    :return: Generator with data of form (invols[m,f, hyperparam], outvols[[m,w],w], segvols[m,f], kps[m,f])

    """
    while True:
        inputs, outputs, seg_maps = next(base_generator)
        if ranges is None:
            hyp = np.stack([[random_hyperparam() for _ in range(nb_classes)] for _ in range(batch_size)])
        else:
            hyp = np.stack([[random_hyperparam_in_range(ranges) for _ in range(nb_classes)] for _ in range(batch_size)]) # of shape:

        inputs = (*inputs, hyp)

        weight_tensors=list()
        for i in np.arange(inputs[0].shape[0]):
            hyp_tensor= utils.set_weight_tensor_per_class(outputs[1][np.newaxis, i], hyp[np.newaxis, i])
            weight_tensors.append(hyp_tensor)
        weight_tensors=np.concatenate(weight_tensors, axis=0)

        outputs=(np.concatenate([outputs[0], weight_tensors], axis=-1), weight_tensors)
        yield inputs, outputs


def hyp_generator_sa_vector_val(base_generator: Generator, return_mode: int, val_hyp: list) -> Generator:
    """
    Generator for validation of Hypermorph.
    inputs = [x, y, val_hyp]
    outputs = [y, hyp_tensor]
    # outputs = [y, hyp_tensor, img_mask]

    :param spatially_adaptive:
    :param val_hyp: either float or weight list with weight per class (spatially-adaptive case)
    :param return_mode: of validation/test dataset
    :param batch_size: batch size
    :param base_generator: the base generator for the desired dataset
    :return: Generator with data of form (invols[m,f, hyperparam], outvols[m,f], segvols[m,f], kps[m,f]) OR
                                        (invols[m,f, hyp_tensor], outvols[m,f], segvols[m,f], kps[m,f]) with hyp based on seg_map of moving image
    """
    assert return_mode in [4,6]
    val_hyp = np.array(val_hyp)[np.newaxis, ...]

    while True:
        if return_mode == 4:
            inputs, outputs, segs = next(base_generator)
        elif return_mode == 6:
            inputs, outputs, segs, kps = next(base_generator)
        else:
            print("Wrong return mode in hyp_generator_val()!", return_mode)

        inputs = (*inputs, val_hyp)
        outputs = None

        if return_mode == 4:
            yield (inputs, outputs, segs)
        else:
            yield (inputs, outputs, segs, kps)


def hyp_generator_elastic(batch_size: int, base_generator: Generator, constrained=False):
    """
    Generator for training of linear elastic Hypermorph.
    inputs = [x, y, hyp]
    outputs = [[y, img_mask], img_mask]

    :param batch_size: batch size
    :param base_generator: data generator
    :param ranges: if desired, predefined ranges for the Lame parameters mu and lambda: [[lambda_min, lambda_max],[mu_min, mu_max]]
    :return: Generator with data of form (invols[m,f, lambda, mu], outvols[m,f])
    """
    while True:
        inputs, outputs = next(base_generator)
        if constrained:
            lam = [random_hyperparam() for _ in range(batch_size)]
            mu = [random_hyperparam_in_range([0, 1 - lam[b]])for b in range(batch_size)]  # constrain elastic params to max sum up to 1
        else:
            lam = [random_hyperparam() for _ in range(batch_size)]
            mu = [random_hyperparam() for b in range(batch_size)]

        hyp = np.stack([lam, mu], axis=-1)

        inputs = (*inputs, hyp)  # outputs: [y, zeros]
        yield (inputs, outputs)


def hyp_generator_elastic_sa_vector(batch_size: int, base_generator: Generator, nb_classes: int, constrained=False):
    """
    Generator for training of linear elastic Hypermorph.
    inputs: [x,y,hyp]
    outputs: [[y, seg_x, img_mask], [seg_x, img_mask]]
    outputs: [[y, wlam, wmu, img_mask], [wlam, wmu, img_mask]]

    :param batch_size: batch size
    :param base_generator: data generator
    :param ranges: if desired, predefined ranges for the Lame parameters mu and lambda: [[lambda_min, lambda_max],[mu_min, mu_max]]
    :return: Generator with data of form (invols[m,f, lambda, mu], outvols[m,f])
    """

    while True:
        inputs, outputs, segs = next(base_generator)
        if constrained:
            hyp1 = np.stack([[random_hyperparam() for _ in range(nb_classes)] for _ in range(batch_size)]) # [bs, classes], lambda
            hyp2 = np.stack([[random_hyperparam_in_range([0,1-hyp1[b,c]]) for c in range(nb_classes)] for b in range(batch_size)]) # [bs, classes], mu
        else:
            hyp1 = np.stack([[random_hyperparam() for _ in range(nb_classes)] for _ in range(batch_size)]) # [bs, classes], lambda
            hyp2 = np.stack([[random_hyperparam() for _ in range(nb_classes)] for _ in range(batch_size)]) # [bs, classes], mu
        hyp=np.concatenate([hyp1, hyp2], axis=-1) # [bs,c*2]

        weight_tensors_lam = list()
        weight_tensors_mu = list()
        for i in np.arange(inputs[0].shape[0]):
            tmp_lam = utils.set_weight_tensor_per_class(outputs[1][np.newaxis, i], hyp1[np.newaxis, i])
            tmp_mu = utils.set_weight_tensor_per_class(outputs[1][np.newaxis, i], hyp2[np.newaxis, i])
            weight_tensors_lam.append(tmp_lam)
            weight_tensors_mu.append(tmp_mu)

        weight_tensors_lam = np.concatenate(weight_tensors_lam, axis=0)
        weight_tensors_mu = np.concatenate(weight_tensors_mu, axis=0)

        inputs = (*inputs, hyp)
        outputs = (np.concatenate([outputs[0], weight_tensors_lam + weight_tensors_mu], axis=-1),
                   np.concatenate([weight_tensors_lam, weight_tensors_mu], axis=-1))
        yield (inputs, outputs)


def warp_kps(warp, fix_lms):
    fix_lms_disp_x = map_coordinates(warp[:, :, :, 0], fix_lms.transpose())
    fix_lms_disp_y = map_coordinates(warp[:, :, :, 1], fix_lms.transpose())
    fix_lms_disp_z = map_coordinates(warp[:, :, :, 2], fix_lms.transpose())
    fix_lms_disp = np.array((fix_lms_disp_x, fix_lms_disp_y, fix_lms_disp_z)).transpose()

    kps_pred = fix_lms + fix_lms_disp
    return kps_pred


def set_weight_tensor_per_class(seg_mask: np.array, weight_ls : np.array)-> np.array:
    """
    Inserts the class weights at the respective locations in the segmentation map tensor
    :param seg_mask: segmenatation mask
    :param weight_ls: list of regularization weights, one per class in ascending order (i.e. [class0_w, class1_w, class2_w, ...)
    :return: np.array of same size as seg_mask, with class specific weights in segmentation areas
    """

    assert seg_mask.dtype == np.int16, seg_mask.dtype
    assert np.unique(seg_mask).shape[0] <= weight_ls.shape[1], str(np.unique(seg_mask).shape)+"   "+ str(weight_ls.shape)

    weight_tensor= np.zeros(seg_mask.shape)
    for batch in range(seg_mask.shape[0]):
        for c, w in enumerate(weight_ls[batch]):
            np.putmask(weight_tensor[batch], seg_mask[batch] == c, w)

    return weight_tensor

