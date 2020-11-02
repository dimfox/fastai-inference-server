
from pathlib import Path
import io
import datetime
import logging
import time

from fastai.data import block, transforms
from fastai.vision import core as vision_core
from fastai.vision.data import ImageBlock
from fastai.vision.augment import Resize
from fastai.vision.learner import cnn_learner
from torchvision.models.resnet import resnet18

from fastai.vision.core import PILImage
import PIL


def timeit(method):
  def timed(*args, **kw):
    ts = time.time()
    result = method(*args, **kw)
    te = time.time()
    name = kw.get('log_name', method.__name__.upper())
    logging.debug('[%s]: %fms', name, (te-ts)*1000)
    return result
  return timed


def load_inf_model(model_file:str, data_path:str):
  """Creates an inference model from a pth file.

  model_file: name of the model, without directory and suffix. E.g.
      load_inf_model('bears') will try to load saved model from
      models/bears.pth
  data_path: directory that contains data used for training the model.
      To build a inference model, we only need the directory structure
      and some data in each directory.
  """
  bears = block.DataBlock(
      blocks=(ImageBlock, block.CategoryBlock), 
      get_items=transforms.get_image_files, 
      splitter=transforms.RandomSplitter(valid_pct=0.2, seed=42),
      get_y=transforms.parent_label,
      item_tfms=Resize(224))
  dls = bears.dataloaders(data_path)
  # resnet18 must matches the model we used when training
  learn = cnn_learner(dls, resnet18)
  learn.load(model_file)
  return learn


def predict_pil_img(learner, img:PIL.Image):
  """Use the learner to predict category of the image.

  learner: a cnn_learner
  img: An instance of PIL image.
  """
  # Convert PIL Image object to bytes
  output = io.BytesIO()
  img.save(output, format='PNG')
  return predict_bytes(learner, output)


@timeit
def predict_bytes(learner, io_data:io.BytesIO):
  """Use the learner to predict category of the image.

  learner: a cnn_learner
  io_data: an instance of io.BytesIO that contains an image.
  """
  # predict function can take a PILImage
  #   return learner.predict(vision_core.PILImage.create(hex_data))
  # or just the byte array
  result = learner.predict(io_data.getvalue())
  predict_idx = result[1].int()
  return dict(
      predict_class=result[0],
      prob=result[2].tolist()[predict_idx])


if __name__ == '__main__':
  learn_inf = load_inf_model('bears3', 'data')
  img_file = sys.argv[1]
  print(predict_pil_img(learn_inf, PIL.Image.open(img_file)))