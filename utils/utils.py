import logging
import os

from ml_serving.drivers import driver


LOG = logging.getLogger(__name__)


template = """
<!DOCTYPE html>
<html>
<head>
<style>
table, th, td {
  border: 1px solid black;
  border-collapse: collapse;
}
th, td {
  padding: 7px;
}
table tr:nth-child(even) {
  background-color: #eee;
}
table tr:nth-child(odd) {
 background-color: #fff;
}
body {
font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Noto Sans", "Ubuntu", "Droid Sans", "Helvetica Neue", sans-serif;
}
</style>
</head>
<body>
<h1 style="text-align:center">Top 100 Faces</h1>
<table style="width:100%">
  <tr>
    <th>##</th>
    <th>Image</th> 
    <th>Max Duration</th>
    <th>Name</th>
    <th>Time intervals</th>
  </tr>
  {% for face_info in data %}
  <tr>
      <td>{{ loop.index }}</td>
      <td>
      {% for img in face_info[2] %}
        <img src="data:image/jpeg;base64,{{ img }}"/>
      {% endfor %}
      </td>
      <td>{{ '%0.2f' | format(face_info[4]) }} sec</td>
      <td>{{ face_info[0] }}</td>
      <td>{{ face_info[3] }}</td>
  </tr>
  {% endfor %}
</table>
</body>
</html>
"""


def get_driver(model_path: str, description: str = None, driver_name: str = None, **kwargs) -> driver.ServingDriver:
    path_device = model_path.split(',')
    if len(path_device) > 1:
        kwargs['device'] = path_device[1]
        model_path = path_device[0]

    if not driver_name:
        # detect tensorflow or openvino model
        if os.path.isdir(model_path) and os.path.isfile(os.path.join(model_path, 'saved_model.pb')):
            driver_name = 'tensorflow'
        elif model_path.endswith('.pth') or model_path.endswith('.pkl'):
            # pytorch
            driver_name = 'pytorch'
        else:
            driver_name = 'openvino'
    if description:
        LOG.info("Load %s %s model from %s..." % (description, driver_name, model_path))
    else:
        LOG.info("Load undescribed %s model from %s..." % (driver_name, model_path))
    drv = driver.load_driver(driver_name)
    d = drv()
    d.load_model(model_path, **kwargs)
    return d


def configure_logging():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-5s %(name)-10s [-] %(message)s',
        level='INFO'
    )
    logging.root.setLevel(logging.INFO)
