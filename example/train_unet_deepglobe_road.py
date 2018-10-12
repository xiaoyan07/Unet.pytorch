import torch
import logging
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_utils
from module.unet import DeepUnet
from data.seg_data import SegData
from module.loss import BinarySegmentationLoss
from util.train import TrainPack

cudnn.benchmark = True
cudnn.enabled = True
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 0. config
image_dir = '../dev/road_data'
mask_dir = '../dev/road_data'
model_dir = './log'

weight_decay = 5e-4
momentum = 0.9
base_lr = 0.1
batch_size = 4
num_epochs = 100
save_per_steps = 1000

# 1. prepare data
train_data = SegData(image_dir=image_dir,
                     mask_dir=mask_dir,
                     training=True)
val_data = SegData(image_dir=image_dir,
                   mask_dir=mask_dir,
                   training=False)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
# 2. modeling
model = DeepUnet(num_classes=1,
                 use_softmax=False,
                 encoder_batchnorm=False,
                 decoder_batchnorm=False)
model.to(device)
model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
# 3. optimize
bs_loss = BinarySegmentationLoss()
trainable_parameters = [p for p in model.parameters() if p.requires_grad]
frozen_parameters = [name for name, p in model.named_parameters() if not p.requires_grad]
for name in frozen_parameters:
    logger.info('{} is frozen.'.format(name))

optimizer = torch.optim.SGD(trainable_parameters, momentum=momentum,
                            lr=base_lr, weight_decay=weight_decay)
lr_schedule = lr_utils.CosineAnnealingLR(optimizer, num_epochs, 1e-7)

TrainPack(
    model_dir=model_dir,
    model=model,
    loss=bs_loss,
    optimizer=optimizer,
    lr_schedule=lr_schedule,
    save_per_steps=save_per_steps
)
