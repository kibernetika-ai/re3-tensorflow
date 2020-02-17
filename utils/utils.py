import logging
import os

from ml_serving.drivers import driver


LOG = logging.getLogger(__name__)


def get_driver(model_path: str, description: str = None, driver_name: str = None, **kwargs) -> driver.ServingDriver:
    path_device = model_path.split(',')
    if len(path_device) > 1:
        kwargs['device'] = path_device[1]
        model_path = path_device[0]

    if not driver_name:
        # detect tensorflow or openvino model
        if os.path.isdir(model_path) and os.path.isfile(os.path.join(model_path, 'saved_model.pb')):
            driver_name = 'tensorflow'
        elif model_path.endswith('.pth'):
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