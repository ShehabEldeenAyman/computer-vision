#author: shehab el deen ayman mounir
DATA_DIR = 'data'
DATA_SET = 'data_set'
WEIGHTS_FILE = 'yolo_weights.ckpt'

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus','car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

ANCHOR = [0.57273, 1.87446, 3.33843, 7.88282, 9.77052, 0.677385, 2.06253, 5.47434, 3.52778, 9.16828]

GPU = ''
IMAGE_SIZE = 416   
LEARN_RATE = 0.0001   
MAX_ITER = 20000    
SUMMARY_ITER = 5    
SAVER_ITER = 50    
BOX_PRE_CELL = 5    
CELL_SIZE = 13      
BATCH_SIZE = 32     
ALPHA = 0.1
THRESHOLD = 0.3   
