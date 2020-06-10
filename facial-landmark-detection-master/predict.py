from my_cascade import Cascador
from data_load import TrainSet
from shape import Shape
import numpy as np
import cv2


def display(img, gtPnts, resPnts):
    gtPnts = np.round(gtPnts).astype(np.int32)
    resPnts = np.round(resPnts).astype(np.int32)

    showImg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(gtPnts.shape[0]):
        cv2.circle(showImg, (gtPnts[i, 0], gtPnts[i, 1]),
                   3, (0, 0, 255), -1)
        cv2.circle(showImg, (resPnts[i, 0], resPnts[i, 1]),
                   3, (255, 0, 0), -1)
    return showImg

model = './model/train.model'
cas = Cascador()
cas = cas.loadModel(model)

folders = ['data/I/', 'data/II/']
imgListPath = folders[1] + 'label.txt'
pathList = open(imgListPath, 'r').readlines()
reader = TrainSet()

for imgPath in pathList:
    img, gtShape, bndBox = reader.read(imgPath, folders[1])
    scale = 3
    cropB, img = reader.cropRegion(bndBox, scale, img)
    gtShape = np.subtract(gtShape, (cropB[0], cropB[1]))
    # TODO try face detector.
    bndbox = Shape.getBBoxByPts(gtShape)
    # Set the initial shape
    # print('bndbox: {}'.format(bndbox))
    # print('type bndbox: {}'.format(type(bndbox)))
    initShape = Shape.shapeNorm2Real(cas.meanShape, bndbox)
    # Detect the landmark
    cas.detect(img, bndbox, initShape)
    showImg = display(img, gtShape, initShape)
    cv2.imshow("Landmark", showImg)
    key = cv2.waitKey(1000)
    if key in [ord("q"), 27]:
        break

cv2.destroyAllWindows()
