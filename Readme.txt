---WARNING!!!---
the 'live_cap' and 'temp' modules will not run without the presence of an Nvidia GPU

important notes:
the project is divided into two working modules: live_cap.py (for cars and pedestrians) and lane detectionfinal.py (for lanes)
there is a third file (temp.py) that combines both modules and can be ran instead of running the two modules seperately


to run the modules:
open command prompt by typing "cmd"
type python + "module name" + .py
example: python live_cap.py

software requirements:
anaconda prompt
spyder ide
Nvidia CudaToolkit 9
Nvidia Cudnn 7

library requirements:
TensorGPU (not the regular tensor)
openCV
numpy
scikit learn

