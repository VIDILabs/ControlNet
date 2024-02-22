from share import *

import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from illumination_dataset import IlluminationGridDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# Configs
base_checkpoint = './models/control_sd15_ini.ckpt'
resume_checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(base_checkpoint, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = IlluminationGridDataset()
dataloader = DataLoader(dataset, num_workers=88, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], resume_from_checkpoint=resume_checkpoint)


# Train!
trainer.fit(model, dataloader)
