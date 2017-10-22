
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap, QTransform

BIRD_IMAGES = ('assets/sprites/redbird-upflap.png',
               'assets/sprites/redbird-midflap.png',
               'assets/sprites/redbird-downflap.png')

BACKGROUD_IMAGE = 'assets/sprites/background-black.png'

BASE_IMAGE = 'assets/sprites/base.png'

PIPE_IMAGE = 'assets/sprites/pipe-green.png'

NUM_IMAGES = ('assets/sprites/0.png',
              'assets/sprites/1.png',
              'assets/sprites/2.png',
              'assets/sprites/3.png',
              'assets/sprites/4.png',
              'assets/sprites/5.png',
              'assets/sprites/6.png',
              'assets/sprites/7.png',
              'assets/sprites/8.png',
              'assets/sprites/9.png')

def getHitmask(images, flip=False):
    img = plt.imread(images)
    if flip:
        img = img[::-1,:]
    mask = img[:,:,3]
    pos = np.arange(mask.shape[1])+((np.arange(mask.shape[0])*1000).reshape(-1,1))
    return pos*mask 

def loadResource():
    PIXMAP = {}
    PIXMAP['bird'] = [QPixmap(img) for img in BIRD_IMAGES]
    PIXMAP['bg'] = QPixmap(BACKGROUD_IMAGE)
    PIXMAP['base'] = QPixmap(BASE_IMAGE)
    PIXMAP['pipe'] = [QPixmap(PIPE_IMAGE), QPixmap(PIPE_IMAGE).transformed(QTransform().rotate(180))]

    HITMASK = {}
    HITMASK['bird'] = [getHitmask(img) for img in BIRD_IMAGES]
    HITMASK['pipe'] = [getHitmask(PIPE_IMAGE), getHitmask(PIPE_IMAGE, flip=True)]
    return PIXMAP, HITMASK
