import time
import numpy as np
from ml_serving.drivers import driver
drv = driver.load_driver('openvino')
serving = drv()
#serving.load_model('openvino/re3.xml')
serving.load_model('./models/re3/openvino/FP16/re3.xml',flexible_batch_size=False,device='MYRIAD')
input_name = list(serving.inputs.keys())
print(input_name)
print(list(serving.outputs.keys()))


summ = 0
NUM = 1000
for i in range(NUM):
    input1 = np.random.randn(1, 3,227, 227)
    input2 = np.random.randn(1, 3, 227, 227)
    start = time.time()
    outputs = serving.predict({'Placeholder': input1,'Placeholder_1': input2})
    out = outputs['re3/fc6/Relu']
    #print(out.shape)
    duration = time.time() - start
    summ += duration
print(f'{NUM} inferences for {summ:.3f} sec')
print(f'Average time per inference: {summ / NUM * 1000:.3f} msec')