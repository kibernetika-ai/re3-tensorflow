import time
import numpy as np
from ml_serving.drivers import driver
drv = driver.load_driver('tensorflow')
serving = drv()
#serving.load_model('openvino/re3.xml')
serving.load_model('./models/re3/re3.pb',inputs='Placeholder:0,Placeholder_1:0', outputs='re3/fc6/Relu:0')
input_name = list(serving.inputs.keys())
print(input_name)
print(list(serving.outputs.keys()))


summ = 0
NUM = 1000
for i in range(NUM):
    input1 = np.random.randn(1,227, 227,3)
    input2 = np.random.randn(1, 227, 227,3)
    start = time.time()
    outputs = serving.predict({'Placeholder:0': input1,'Placeholder_1:0': input2})
    out = outputs['re3/fc6/Relu:0']
    #print(out.shape)
    duration = time.time() - start
    summ += duration
print(f'{NUM} inferences for {summ:.3f} sec')
print(f'Average time per inference: {summ / NUM * 1000:.3f} msec')