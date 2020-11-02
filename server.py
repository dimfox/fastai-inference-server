import io
import json
import sys
import aiohttp
import asyncio
import uvicorn
import base64
from starlette import responses as sl_res
from starlette import applications as sl_app
from starlette.middleware import cors
import PIL

from lib import util


DATA_DIR = 'data'
MODEL_NAME = 'bears'


app = sl_app.Starlette()
app.add_middleware(cors.CORSMiddleware,
                   allow_origins=['*'],
                   allow_headers=['X-Requested-With',
                   'Content-Type'])


async def setup_learner():
    learner = util.load_inf_model(MODEL_NAME, DATA_DIR)
    return learner


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learner = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/healthcheck', methods=['GET'])
def status(request):
    return sl_res.JSONResponse(dict(status='OK'))

@app.route('/analyze:predict', methods=['POST'])
async def analyze(request):
    data = await request.body()
    img_data = json.loads(data.decode('utf-8'))['data']
    img_bytes = base64.b64decode(img_data)
    try:
      return sl_res.JSONResponse(dict(
          prediction=util.predict_bytes(learner, io.BytesIO(img_bytes))))
    except PIL.UnidentifiedImageError:
      return sl_res.JSONResponse(dict(
          error='Cannot read image'))
 

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=8501, log_level="info")