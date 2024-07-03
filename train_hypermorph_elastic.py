"""
Adaptation of official example script for training a HyperMorph model to tune the
regularization weight hyperparameter.
"""
import argparse
import datetime
import os

import SimpleITK as sitk  # to suppress warnings

import datasets
import utils
from regularizers import elastic_regularizer_modersitzki_tf

sitk.ProcessObject_SetGlobalWarningDisplay(False)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf

import voxelmorph as vxm
import hypermorph_spatially_adaptive as vxm_sa


def image_loss(y_true, y_pred):
    if ndim == 2:
        lam = model.references.hyper_val[..., 0, np.newaxis, np.newaxis]  # global lambda
        mu = model.references.hyper_val[..., 1, np.newaxis, np.newaxis]  # global mu
    elif ndim == 3:
        lam = model.references.hyper_val[..., 0, np.newaxis, np.newaxis, np.newaxis]  # global lambda
        mu = model.references.hyper_val[..., 1, np.newaxis, np.newaxis, np.newaxis]  # global mu

    hyp = (1 - lam - mu)  # mu + lambda <=1
    y_true = y_true[..., 0, np.newaxis]

    if crop_in_loss:
        mask = tf.equal(y_true, 0)
        y_pred = tf.where(mask, tf.zeros_like(y_pred), y_pred)
    return hyp * image_loss_func(y_true, y_pred, img_mask=None)


def grad_loss(y_true, y_pred):
    if ndim == 2:
        lam = model.references.hyper_val[..., 0, np.newaxis, np.newaxis]  # global lambda
        mu = model.references.hyper_val[..., 1, np.newaxis, np.newaxis]  # global mu
    elif ndim == 3:
        lam = model.references.hyper_val[..., 0, np.newaxis, np.newaxis, np.newaxis]  # global lambda
        mu = model.references.hyper_val[..., 1, np.newaxis, np.newaxis, np.newaxis]  # global mu

    elastic_loss = elastic_regularizer_modersitzki_tf(y_pred, lam, mu, img_mask=None, reduce=True, average=True)
    return elastic_loss


# parse the commandline
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, dest="dataset", help="dataset")

parser.add_argument("--modelname", type=str, dest="modelname", help="modelname")

parser.add_argument("--epochs", type=int, dest="epochs", help="epochs")

parser.add_argument("--bs", type=int, dest="batch_size", help="bs")

parser.add_argument("--regn", type=int, dest="regn", default=11, help="regn")

args = parser.parse_args()

dataset = args.dataset
model_name = args.modelname
data_base_path = "datasets"
args.model_dir = "results/elastic-hypermorph/" + dataset + "/" + model_name

args.image_loss = "ncc"
args.lr = 1e-4
val_interval = 10
val_hyp = 0.1
load_model_str = None

if dataset == "LungCTCT":  # lung CT inh/exh
    datapath = data_base_path + "/LungCT"
    train_dataset = datasets.Learn2RegLungCTDataset(datapath, mode="train", normalize_mode=True)
    val_dataset = datasets.Learn2RegLungCTDataset(datapath, mode="val2", normalize_mode=True)
    use_keypoints = True
    crop_in_loss = True
if dataset == "LungCTCT-2d":  # lung CT inh/exh
    datapath = data_base_path + "/LungCT"
    train_dataset = datasets.Learn2RegLungCTDataset(datapath, mode="train", normalize_mode=True, dim_mode='2d')
    val_dataset = datasets.Learn2RegLungCTDataset(datapath, mode="val2", normalize_mode=True, dim_mode='2d')
    use_keypoints = False
    crop_in_loss = True
elif dataset == "NLST23":
    datapath = data_base_path + "/NLST23/NLST"
    train_dataset = datasets.NLST2023Dataset(datapath, mode="train", normalize_mode=True)
    val_dataset = datasets.NLST2023Dataset(datapath, mode="val2", normalize_mode=True)
    use_keypoints = True
    crop_in_loss = True
elif dataset == "NLST23-2d":
    # datapath = data_base_path + "/NLST23/NLST_preprocessed_2d"
    datapath = data_base_path + "/NLST23/NLST"
    train_dataset = datasets.NLST2023Dataset(datapath, mode="train", normalize_mode=True, dim_mode='2d')
    val_dataset = datasets.NLST2023Dataset(datapath, mode="val2", normalize_mode=True, dim_mode='2d')
    use_keypoints = False
    crop_in_loss = True
elif dataset == "ACDC":
    datapath = data_base_path + "/ACDC"
    train_dataset = datasets.ACDCDataset(datapath, mode="train", normalize_mode=True, roi_only=True,
                                         dim_mode='2d-middle')
    val_dataset = datasets.ACDCDataset(datapath, mode="val2", normalize_mode=True, roi_only=True, dim_mode='2d-middle')
    use_keypoints = False
    crop_in_loss = False
else:
    print("Wrong dataset name specified!")
print("Using data ", dataset, " of length (train) ", len(train_dataset), " and (val) ", len(val_dataset),
      " and batchsize ", args.batch_size)
print("GPU available? ", tf.config.list_physical_devices('GPU'))

# logging stuff
log_dir = args.model_dir + "/log"
log_file_train = args.model_dir + "/log/train.txt"
log_file_val = args.model_dir + "/log/val.txt"

if not os.path.isdir(log_dir):
    os.makedirs(log_dir, exist_ok=True)

with open(log_file_train, 'w') as f:
    f.write(model_name + " -- " + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + "\n")
with open(log_file_val, 'w') as f:
    f.write(model_name + " -- " + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + "\n")

base_generator = utils.scan_to_scan_generator(train_dataset, batch_size=args.batch_size)
val_base_generator = utils.scan_to_scan_generator(val_dataset, batch_size=args.batch_size)
train_generator = utils.hyp_generator_elastic(args.batch_size, base_generator, constrained=True)
val_generator = utils.hyp_generator_elastic(args.batch_size, val_base_generator, constrained=True)

args.steps_per_epoch = len(train_dataset) / args.batch_size
val_steps_per_epoch = len(val_dataset) / args.batch_size

ndim = len(train_dataset.imgshape)
assert ndim == 2 or ndim == 3

if ndim == 3:
    sample_shape = (args.batch_size, train_dataset.imgshape[0], train_dataset.imgshape[1], train_dataset.imgshape[2], 1)
else:
    sample_shape = (args.batch_size, train_dataset.imgshape[0], train_dataset.imgshape[1], 1)

inshape = sample_shape[1:-1]
nfeats = 1
print(sample_shape, inshape, nfeats)

# training parameters
nb_features = [[32, 32, 32, 32],  # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]
unet_input_features = 2
loss_name = args.image_loss
int_steps = 7
int_downsize = 1
image_sigma = 0.05  # estimated image noise for mse image scaling
initial_epoch = 0

print("GPU available? ", tf.config.list_physical_devices('GPU'))
device = 'cuda'

# unet architecture
enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]

# prepare model checkpoint save path
save_filename = os.path.join(args.model_dir, '{epoch:04d}.h5')

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device('GPU')

# build the model
model = vxm.networks.HyperVxmDense(inshape=inshape, nb_unet_features=[enc_nf, dec_nf], int_steps=int_steps,
    int_resolution=int_downsize, src_feats=nfeats, trg_feats=nfeats, svf_resolution=1, nb_hyp_params=2)

if load_model_str:
    model = vxm.networks.HyperVxmDense.load(load_model_str, input_model=None)

# prepare image loss
if loss_name == 'ncc':
    image_loss_func = vxm_sa.losses.NCC().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % loss_name)

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=args.lr), loss=[image_loss, grad_loss])
save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename, save_freq=5000)

model.fit(train_generator, initial_epoch=initial_epoch, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
          validation_data=val_generator, validation_freq=val_interval, validation_steps=val_steps_per_epoch,
          callbacks=[save_callback], verbose=1)

# save final weights
model_path = os.path.join(args.model_dir, 'model_final.h5')
model.save(model_path)