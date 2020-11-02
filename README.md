# fastai-inference-server
A python server that service fast.ai inference model.

## Download model
The model is downloaded from fast.ai's 02_production notebook. Execute the following in the notebook:

```
  learner.save('bears')
```
This will save a `models/bears.pth`, which can be downloaded from notebook.

This repo provides a starlette sever that can load this model and categorize a picture with the mode.

## Create directories
We should not need any data to do inference. But because of the way fast.ai cnn_learner is initialized, 
we must provide directories, one for each category, and each direcotry must have some data. If the model
is to classy pictures to three types of bears: `black_bear`, `grizzly_bear`, `teddy_bear`, we need three
directories. One for each category, the directory name will be used for class name. Each directory must
contain some files.

If you already have a Azure search key, you can build with the utility. Otherwise, it might be easier to
create the data manually.
```
python
>>> from lib import init_model_data
/usr/local/lib/python3.8/site-packages/fastcore/foundation.py:52: UserWarning: `patch_property` is deprecated and will be removed; use `patch(as_prop=True)` instead
  warnings.warn("`patch_property` is deprecated and will be removed; use `patch(as_prop=True)` instead")
>>> init_model_data.build_data(categories=('black_bear', 'grizzly_bear', 'teddy_bear'), data_path='data', azure_search_key="xxxxx")
```

## Start server
We can start the server directly
```
pip3 install -r requirements.txt
python3 server.py serve
```

Or we can use docker
```
docker build -f Dockerfile  -t fastai-serving .
#docker run --rm -p 8501:8501 -t fastai-serving .
```

## Inference
To test the server with curl:
```
(echo -n '{"data" : "'"$( base64 ~/Downloads/r1.jpg)"'"}') | curl -X POST -H "Content-Type: application/json" -d @- localhost:8501/analyze:predict
```
