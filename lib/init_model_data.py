"""Initialize model data.

Fastai's cnn_learner need directories on disk to know the classification
of the model. And each directory must have some files in it.
"""
from pathlib import Path

from fastai.vision import utils as v_utils
from fastcore.foundation import L

from azure.cognitiveservices.search.imagesearch import ImageSearchClient as api
from msrest.authentication import CognitiveServicesCredentials as auth


AZURE_SEARCH_KEY = 'xxxxx'
CATEGORIES = ('grizzly_bear', 'black_bear', 'teddy_bear')

def _search_images_bing(key, term, min_sz=128):
  client = api('https://api.cognitive.microsoft.com', auth(key))
  return L(client.images.search(
      query=term, count=20,
      min_height=min_sz, min_width=min_sz).value)


def build_data(categories, data_path, azure_search_key):
  """Creates a parent directory `data_path`.
     Creates one directory for each category and download some images
     into that directory.
  """
  path = Path(data_path)
  if not path.exists():
    path.mkdir()
    for o in categories:
      dest = (path/o)
      dest.mkdir(exist_ok=True)
      results = _search_images_bing(azure_search_key, o)
      v_utils.download_images(dest, urls=results.attrgot('content_url'))