from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.potsdam_dataset import *
from geoseg.models.FTUNetFormer import ft_unetformer
from geoseg.scheduler.CosineAnnealingWithWarmup import CosineAnnealingWarmupRestarts
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 100
ignore_index = len(CLASSES)
train_batch_size = 4
val_batch_size = 4
lr = 1e-3
weight_decay = 1e-4
backbone_lr = 1e-4
backbone_weight_decay = 1e-4
num_classes = len(CLASSES)
classes = CLASSES
class_weights = torch.tensor([0.5, 1.0])

weights_name = "ftunetformer-aug-512"
weights_path = "model_weights/buildingsegmentation/{}".format(weights_name)
test_weights_name = "ftunetformer-aug-512"
log_name = 'buildingsegmentation/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = ft_unetformer(num_classes=num_classes, decoder_channels=256)

# define the loss
loss = JointLoss(CrossEntropyLoss(label_smoothing=0.05, ignore_index=ignore_index, weight=class_weights),
                 LovaszLoss(ignore=ignore_index), 0.8, 0.2)

use_aux_loss = False

# define the dataloader

train_dataset = PotsdamDataset(data_root='data',img_dir='images/train',mask_dir='annotations/train', mode='train',
                               mosaic_ratio=0.25, transform=train_aug)

val_dataset = PotsdamDataset(data_root='data',img_dir='images/val',mask_dir='annotations/val', transform=val_aug)
test_dataset = PotsdamDataset(data_root='data',
                            img_dir='images/val',mask_dir='annotations/val',
                              transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=20, warmup_steps=5, gamma=.5, min_lr=1e-6, cycle_mult=1, max_lr=lr)