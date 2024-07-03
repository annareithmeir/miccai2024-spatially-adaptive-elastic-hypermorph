import numpy as np
import scipy
from typing import Optional

def dice(image1 : np.ndarray, image2 : np.ndarray, img_mask:Optional[np.ndarray]=None) -> float:
    """
    Taken from github of conditional LapIRN by Tony Mok
    :param image1: pred
    :param image2:true
    :return:
    """
    unique_class = np.unique(image2)
    dice = 0
    num_count = 0

    if img_mask is not None:
        image1[img_mask==0]=0
        # image2[img_mask==0]=0

    for i in unique_class:
        if (i == 0) or ((image1 == i).sum() == 0) or ((image2 == i).sum() == 0):
            continue

        sub_dice = np.sum(image2[image1 == i] == i) * 2.0 / (np.sum(image1 == i) + np.sum(image2 == i))
        dice += sub_dice
        num_count += 1
    if num_count==0:
        return 0
    else:
        return dice / num_count


def dice_per_class(image1 : np.ndarray, image2 : np.ndarray, classes : list[int], img_mask:Optional[np.ndarray]=None) -> list[float]:
    """
    Computes dice scores per class labels. Based on Tony Mok LapIRN implementation
    :param image1:
    :param image2:
    :param classes: list of labels to compute dice on
    :return: list of dice scores
    """

    dice_ls=[]

    if img_mask is not None:
        image1[img_mask==0]=0
        # image2[img_mask==0]=0
        # fig = plt.figure(figsize=(20, 6))
        # print(image1.shape, image2.shape)
        # image=image1.squeeze()
        # image_size = image.shape
        # slices = [int(image_size[0] / 2), int(image_size[1] / 2), int(image_size[2] / 2)]
        # for a in range(0, 3):
        #     ax = fig.add_subplot(2, 3, a + 1)
        #     if a == 0:
        #         slice = image[slices[a], :, :]
        #
        #     if a == 1:
        #         slice = image[:, slices[a], :]
        #
        #     if a == 2:
        #         slice = image[:, :, slices[a]]
        #
        #     ax.imshow(slice)
        #     # ax.scatter(kp_slice[:, 1], kp_slice[:, 0], marker='x', c='red')
        #     # plt.gca().invert_yaxis()
        #     # plt.colorbar()
        #
        # image=image2.squeeze()
        # image_size = image.shape
        # slices = [int(image_size[0] / 2), int(image_size[1] / 2), int(image_size[2] / 2)]
        # for a in range(0, 3):
        #     ax = fig.add_subplot(2, 3, a + 4)
        #     if a == 0:
        #         slice = image[slices[a], :, :]
        #
        #     if a == 1:
        #         slice = image[:, slices[a], :]
        #
        #     if a == 2:
        #         slice = image[:, :, slices[a]]
        #
        #     ax.imshow(slice)
        #     # ax.scatter(kp_slice[:, 1], kp_slice[:, 0], marker='x', c='red')
        #     # plt.gca().invert_yaxis()
        #     # plt.colorbar()
        #
        # plt.show()

    for i in classes:
        if (i == 0) or (np.sum(image1 == i) == 0) or (np.sum(image2 == i) == 0):
            dice_ls.append(0) #TODO check if correct
            continue

        sub_dice = np.sum(image2[image1 == i] == i) * 2.0 / (np.sum(image1 == i) + np.sum(image2 == i))
        dice_ls.append(sub_dice)
    return dice_ls


def TRE(points1 : np.ndarray, points2 : np.ndarray, spacing : Optional[np.ndarray]=np.array([1,1,1]), squared : Optional[bool]=False) -> float:
    """
    Mean target registration error, i.e. vector L1 norm, of keypoints. Based on Airlab implementation
    :param points1: shape [n,3] or [n,2], corresponding points
    :param points2: shape [n,3] or [n,2], corresponding points
    :param spacing: list of 2 or 3 values, pixel spacing
    :param squared: if true returns L2 norm, else L1
    :return: float
    """
    assert points1.shape[0]==points2.shape[0] , "shapes are "+str(points1[0].shape)+" and "+str(points2[0].shape)
    if not squared:
        return np.sqrt(np.mean(np.linalg.norm((points1 - points2)*spacing, axis=1)))
    else:
        return np.mean(np.linalg.norm(((points1 - points2)*spacing), axis=1))


def warp_keypoints(F_X_Y, fixed_keypoints):
    # input keypoints torch tensors of size [n,3] or [1,n,3]
    F_X_Y_xyz = np.zeros(F_X_Y.shape, dtype=F_X_Y.dtype)
    _, _, x, y, z = F_X_Y.shape
    F_X_Y_xyz[0, 0] = F_X_Y[0, 2] * (x - 1) / 2
    F_X_Y_xyz[0, 1] = F_X_Y[0, 1] * (y - 1) / 2
    F_X_Y_xyz[0, 2] = F_X_Y[0, 0] * (z - 1) / 2

    F_X_Y_xyz_cpu = F_X_Y_xyz[0]
    fixed_keypoints=fixed_keypoints.squeeze()

    fixed_disp_x = scipy.ndimage.map_coordinates(F_X_Y_xyz_cpu[0], fixed_keypoints.transpose())
    fixed_disp_y = scipy.ndimage.map_coordinates(F_X_Y_xyz_cpu[1], fixed_keypoints.transpose())
    fixed_disp_z = scipy.ndimage.map_coordinates(F_X_Y_xyz_cpu[2], fixed_keypoints.transpose())
    lms_fixed_disp = np.array((fixed_disp_x, fixed_disp_y, fixed_disp_z)).transpose()
    warped_fixed_keypoint = fixed_keypoints + lms_fixed_disp

    return warped_fixed_keypoint


# def calculate_tre(F_X_Y, moving_keypoints, fixed_keypoints):
#     # input keypoints torch tensors of size [n,3] or [1,n,3]
#     warped_fixed_keypoint = warp_kp(F_X_Y, moving_keypoints, fixed_keypoints)
#
#     tre_score = TRE(warped_fixed_keypoint, moving_keypoints)
#     return tre_score

# def jacobian_determinant_lapirn(disp):
#     # copied from cLapIRN code
#     J = y_pred + sample_grid
#     dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
#     dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
#     dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]
#
#     Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
#     Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
#     Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])
#
#     Jdet = Jdet0 - Jdet1 + Jdet2
#
#     return Jdet
#
#
# def jacobian_determinant(disp):
#
#     """
#     From voxelmorph code
#     :param y_pred:
#     :param sample_grid:
#     :return:
#
#     jacobian determinant of a displacement field.
#     NB: to compute the spatial gradients, we use np.gradient.
#
#     Parameters:
#         disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
#               where vol_shape is of len nb_dims
#
#     Returns:
#         jacobian determinant (scalar)
#     """
#     import pystrum.pynd.ndutils as nd
#
#     # check inputs
#     volshape = disp.shape[:-1]
#     nb_dims = len(volshape)
#     assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'
#
#     # compute grid
#     grid_lst = nd.volsize2ndgrid(volshape)
#     grid = np.stack(grid_lst, len(volshape))
#
#     # compute gradients
#     J = np.gradient(disp + grid)
#
#     # 3D glow
#     if nb_dims == 3:
#         dx = J[0]
#         dy = J[1]
#         dz = J[2]
#
#         # compute jacobian components
#         Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
#         Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
#         Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])
#         return Jdet0 - Jdet1 + Jdet2
#
#     else:  # must be 2
#         dfdx = J[0]
#         dfdy = J[1]
#         return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
#
#
# def compute_jacobian_matrix(disp: np.array, add_identity:Optional[bool]=False) -> np.ndarray:
#     # from voxelmorph code
#     import pystrum.pynd.ndutils as nd
#
#     # check inputs
#     volshape = disp.shape[:-1]
#     nb_dims = len(volshape)
#     assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'
#
#     # compute grid
#     grid_lst = nd.volsize2ndgrid(volshape)
#     grid = np.stack(grid_lst, len(volshape))
#     # print(grid.shape)
#     # print(grid[...,0])
#     # print(grid[...,1])
#     # print((disp + grid)[...,0])
#     # print((disp + grid)[...,1])
#
#     # compute gradients
#     J = np.gradient(disp + grid) # uses repeated values at borders
#     # print(len(J))
#     # print(J[0].shape)
#     # for i in J:
#         # print(i[...,0])
#         # print(i[...,1])
#      # for x,z,z,3 array, J has 4 elements but should be 3?check!
#     J=np.stack(J, -1)
#     # print(J.shape)
#     # print(J)
#
#     if add_identity: #TODO check
#         for i in np.arange(nb_dims):
#             J[:,:,:, i,i]+=np.ones_like(J[:,:,:, i,i])
#     # print(J[...,:3])
#     return J[...,:3] # ignore last dim
#     # return J
#
#
# def min_max_jacobian_determinant(jac_det, img_mask=None):
#     if img_mask is not None:
#         jac_det[np.where(img_mask==0)]=np.nan
#     return np.nanmin(jac_det), np.nanmax(jac_det)
#
#
# def jacobian_determinant_deviation_from_one(jac_det: np.ndarray, img_mask:Optional[np.ndarray]=None) -> np.ndarray:
#     """
#     mean abs diff between jac det and 1
#     """
#     abs_dev=np.abs(jac_det-1)
#     if img_mask is not None:
#         abs_dev[np.where(img_mask==0)]=np.nan
#     return np.nanmean(abs_dev)
#
#
# def SDlogJ(jac_det, img_mask:Optional[np.ndarray]=None):
#     # std of log of jacdet
#     log_jac_det = np.log(jac_det)
#     if img_mask is not None:
#         log_jac_det[np.where(img_mask==0)]=np.nan
#     return np.nanstd(log_jac_det)
#
#
# def mean_neg_jdet(jac_det, mask:Optional[np.ndarray]=None):
#     if mask is not None:
#         jac_det[np.where(mask==0)]=1
#     if (jac_det < 0).sum() == 0 :
#         return 0
#     else:
#         return (jac_det[jac_det < 0]).mean()
#
#
# def foldings_negative_fraction(jac_det, img_mask=None):
#     if img_mask is not None:
#         jac_det[np.where(img_mask==0)]=1
#     num_foldings = (jac_det < 0).sum()
#     if img_mask is not None:
#         return float(num_foldings) / float(np.count_nonzero(img_mask.flatten()))
#     else:
#         return float(num_foldings)/float(jac_det.size)
#
#
# def hausdorff_distance(y_pred, y_true, p=95, spacing=None):
#     # input needs to be [bs, h, w]
#
#     y_pred=y_pred[np.newaxis, ...]
#     y_true=y_true[np.newaxis, ...]
#     y_pred=torch.from_numpy(y_pred)
#     y_true=torch.from_numpy(y_true)
#     y_pred=torch.movedim(y_pred, -1, 1)
#     y_true=torch.movedim(y_true, -1, 1)
#     assert y_true.shape==y_pred.shape
#     hd=monai.metrics.compute_hausdorff_distance(y_pred, y_true, percentile=p, spacing=spacing)
#     # print(hd, hd.shape)
#
#     return hd.detach().numpy()[0]
#
#
# # https://github.com/vamzimmer/multitask_seg_placenta/blob/main/src/utils/eval_utils.py#L99
# def get_surface_distance_measures(reference, segmentation):
#
#     hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
#     label = 1
#     statistics_image_filter = sitk.StatisticsImageFilter()
#
#     reference_distance_map = sitk.Abs(
#         sitk.SignedMaurerDistanceMap(reference, squaredDistance=False, useImageSpacing=True))
#     reference_surface = sitk.LabelContour(reference)
#     # Get the number of pixels in the reference surface by counting all pixels that are 1.
#     statistics_image_filter.Execute(reference_surface)
#     num_reference_surface_pixels = int(statistics_image_filter.GetSum())
#
#     # Hausdorff distance
#     try:
#         hausdorff_distance_filter.Execute(reference, segmentation)
#         hausdorff = hausdorff_distance_filter.GetHausdorffDistance()
#     except RuntimeError:
#         hausdorff = np.nan
#
#     # surface distances
#     # Symmetric surface distance measures
#     segmented_distance_map = sitk.Abs(
#         sitk.SignedMaurerDistanceMap(segmentation, squaredDistance=False, useImageSpacing=True))
#     segmented_surface = sitk.LabelContour(segmentation)
#
#     # Multiply the binary surface segmentations with the distance maps. The resulting distance
#     # maps contain non-zero values only on the surface (they can also contain zero on the surface)
#     seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
#     ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)
#
#     # Get the number of pixels in the segmented surface by counting all pixels that are 1.
#     statistics_image_filter.Execute(segmented_surface)
#     num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
#
#     # Get all non-zero distances and then add zero distances if required.
#     seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
#     seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
#     seg2ref_distances = seg2ref_distances + \
#                         list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
#     ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
#     ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
#     ref2seg_distances = ref2seg_distances + \
#                         list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
#
#     all_surface_distances = seg2ref_distances + ref2seg_distances
#
#     # Robust hausdorff distance: X% percentile of
#     robust_hausdorff_95 = np.percentile(all_surface_distances, 95)
#
#     # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
#     # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
#     # segmentations, though in our case it is. More on this below.
#     return np.mean(all_surface_distances), robust_hausdorff_95
#     # return hausdorff, 0., 0., 0., 0.
#
#
# def max_shear(u:np.ndarray, mask:Optional[np.ndarray]=None) -> np.ndarray:
#     """
#     See e.g. Imaging of sliding visceral interfaces during breathing (Goksel et al. 2016)
#     :param u: displacement field of shape (x,y,z,3)
#     """
#
#     assert u.shape[-1]==3
#     J= compute_jacobian_matrix(u)
#     # J=tf.convert_to_tensor(J, dtype=tf.float32)
#
#     max_shear = np.empty(u.shape[:3])
#     for i in np.arange(u.shape[0]):
#         for j in np.arange(u.shape[1]):
#             for k in np.arange(u.shape[2]):
#                 # print(i,j,k)
#                 F = J[i, j, k, :, :]
#                 # eigs, _ = tf.linalg.eig(tf.linalg.matmul(tf.transpose(F), F))
#                 eigs, _ = np.linalg.eig(np.matmul(F.T, F))
#                 # max_val = ((tf.math.sqrt(tf.reduce_max(eigs))-tf.math.sqrt(tf.reduce_min(eigs)))*0.5)
#                 max_shear[i, j, k] = (math.sqrt(max(eigs))-math.sqrt(min(eigs)))*0.5
#
#     return max_shear
