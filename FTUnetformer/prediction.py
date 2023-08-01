import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
# from torchvision.transforms import transforms
import albumentations as albu
import ttach as tta


from geoseg.datasets.potsdam_dataset import val_aug
from train_supervision import *
from tools.metric import Evaluator


def rle_encode(mask):
  pixels = mask.flatten()
  pixels = np.concatenate([[0], pixels, [0]])
  runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
  runs[1::2] -= runs[::2]
  return ' '.join(str(x) for x in runs)


class ImageDataset(Dataset):
  def __init__(self, root, transforms=None):
    self.root = root
    self.transforms = transforms
    self.imgs = list(sorted(os.listdir(os.path.join(root, "test_img"))))

  def __getitem__(self, idx):
    # load images and masks
    img_path = os.path.join(self.root, "test_img", self.imgs[idx])
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    img = self.transforms(image=img)['image']
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img
  
  def __len__(self):
    return len(self.imgs)


def prediction():
  config = py2cfg('config/potsdam/ftunetformer.py')
  # 66epoch
  model = Supervision_Train.load_from_checkpoint(os.path.join(config.weights_path, config.test_weights_name+'-v1.ckpt'), config=config)

  model.cuda()
  model.eval()

  transform = albu.Compose([
    albu.Normalize(),
    albu.Resize(256, 256)
  ])
  
  TTA_transform = tta.Compose(
    [
      tta.HorizontalFlip(),
      tta.Rotate90(angles=[0, 90]),
      tta.Scale(scales=[0.75, 1.0, 1.25], interpolation='bicubic', align_corners=False),
    ]
  )

  model = tta.SegmentationTTAWrapper(model, TTA_transform, merge_mode='mean')

  img_test = ImageDataset('data', transform)
  test_loader = torch.utils.data.DataLoader(
    img_test,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=False,
    drop_last=False
  )

  with torch.no_grad():
    with torch.inference_mode():
      result = []
      for images in tqdm(test_loader):
        images = images.float().cuda()
        outputs = model(images)

        # new_outputs = outputs[:,1,:,:] - outputs[:,0,:,:]
        outputs = nn.Softmax(dim=1)(outputs)
        predictions = outputs.argmax(dim=1)
        predictions = predictions.cpu().detach().numpy().astype(np.uint8)
        
        # predictions = torch.sigmoid(outputs).cpu().detach().numpy()
        # predictions = (predictions > 0.5).astype(np.uint8)
        for i in range(len(images)):
          outputs_transform = albu.Resize(224, 224)(image=predictions[i])['image']
          mask_rle = rle_encode(outputs_transform)
          if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
            result.append(-1)
          else:
            result.append(mask_rle)
  
  return result
  
if __name__ == "__main__":
  result = prediction()
  submit = pd.read_csv('./sample_submission.csv')
  submit['mask_rle'] = result
  submit.to_csv('./submit_ftUnetFormer.csv', index=False)