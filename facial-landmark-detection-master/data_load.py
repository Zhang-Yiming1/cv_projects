from PIL import Image
import numpy as np
import re
import math
import copy
from shape import Shape
from affine import Affine
import cv2

class TrainSet(object):
    def __init__(self):
        self.imgDatas = []
        self.gtShapes = []
        self.bndBoxs  = []
        self.initShapes = []
        self.ms2reals  = []
        self.real2mss  = []
        self.meanShape = None
        self.augNum = 1

    def getBBoxByPts(self, pts):
        maxV = np.max(pts, axis=0)
        # print('maxV: {}'.format(maxV))
        minV = np.min(pts, axis=0)
        # print('minV: {}'.format(minV))
        return (minV[0], minV[1],
                maxV[0] - minV[0] + 1,
                maxV[1] - minV[1] + 1)

    def read(self, line, folder):
        gtShape = []
        # Load the ground truth of shape
        line = re.split(r' ', line)
        #rect = line[1:5]
        x = line[5::2]
        y = line[6::2]
        for i in range(len(x)):
            gtShape.append((x[i], y[i]))
        gtShape = np.asarray(gtShape, dtype=np.float32)
        # gtShape = gtShape[0, :, :]
        #print(gtShape.shape)  # (1, 21, 2) --> (21, 2)

        # Load the image data
        img_name = line[0]
        img_path = folder + img_name
        img = Image.open(img_path)
        # gray image
        img = img.convert('L')
        img = np.asarray(img, dtype=np.uint8)
        # print(img)

        # Crop the image
        bndBox = self.getBBoxByPts(gtShape)
        #print('bndBox: {}'.format(bndBox))
        return img, gtShape, bndBox

    def cropRegion(self, bbox, scale, img):
        height, width = img.shape
        w = math.floor(scale * bbox[2])
        h = math.floor(scale * bbox[3])
        x = max(0, math.floor(bbox[0] - (w - bbox[2]) / 2))
        y = max(0, math.floor(bbox[1] - (h - bbox[3]) / 2))
        w = min(width - x, w)
        h = min(height - y, h)

        ### If not use deepcopy, the subImg will hold the whole img's memory
        subImg = copy.deepcopy(img[y:y + h, x:x + w])
        return (x, y, w, h), subImg

    def add(self, img, gtShape, bndBox):
        self.imgDatas.append(img)
        self.gtShapes.append(gtShape)
        self.bndBoxs.append(bndBox)

    def calMeanShape(self):
        meanShape = np.zeros(self.gtShapes[0].shape)
        for i, s in enumerate(self.gtShapes):
            normS = Shape.shapeReal2Norm(s, self.bndBoxs[i])
            meanShape = np.add(meanShape, normS)

        self.meanShape = meanShape / len(self.gtShapes)

    def genTrainData(self, augNum):
        # Step1 : Compute the mean shape
        self.calMeanShape()

        # Set meanshape as the initshape
        for bb in self.bndBoxs:
            initShape = Shape.shapeNorm2Real(self.meanShape,
                                             bb)
            self.initShapes.append(initShape)

        # Translate list into numpy's array
        self.initShapes = np.asarray(self.initShapes,
                                     dtype=np.float32)
        self.gtShapes = np.asarray(self.gtShapes,
                                   dtype=np.float32)
        self.bndBoxs = np.asarray(self.bndBoxs,
                                  dtype=np.float32)

        # Shape augment
        if augNum > 1:
            self.augNum = augNum
            self.initShapes = np.repeat(self.initShapes,
                                        augNum,
                                        axis=0)
            self.gtShapes = np.repeat(self.gtShapes,
                                      augNum,
                                      axis=0)
            self.bndBoxs = np.repeat(self.bndBoxs,
                                     augNum,
                                     axis=0)
            # Permutate the augmented shape
            sampleNum = self.initShapes.shape[0]
            for i in range(sampleNum):
                if 0 == i % sampleNum:
                    continue
                shape = self.initShapes[i]
                self.initShapes[i] = Shape.augment(shape)
        return

    def getAffineT(self):
        num = self.gtShapes.shape[0]
        self.ms2real  = []
        self.real2ms  = []

        for i in range(num):
            ### Project to meanshape coordinary
            bndBox = self.bndBoxs[i]
            initShape = self.initShapes[i]
            mShape = Shape.shapeNorm2Real(self.meanShape,
                                         bndBox)
            T = Affine.fitGeoTrans(initShape, mShape)
            self.real2mss.append(T)
            T = Affine.fitGeoTrans(mShape, initShape)
            self.ms2reals.append(T)

    def calResiduals(self):
        ### Compute the affine matrix
        self.getAffineT()

        self.residuals = np.zeros(self.gtShapes.shape)
        num = self.gtShapes.shape[0]
        for i in range(num):
            # Project to meanshape coordinary
            T = self.real2mss[i]
            bndBox = self.bndBoxs[i]
            err = self.gtShapes[i] - self.initShapes[i]
            err = np.divide(err, (bndBox[2], bndBox[3]))
            err = Affine.transPntsForwardWithSameT(err, T)
            self.residuals[i, :] = err

class DataWrapper(object):
    def __init__(self, para):
        self.path = para['path']  # './data/I/'
        self.augNum = para['augNum']  # 1

    def read(self):
        trainSet = TrainSet()
        # folders = ['data/I/', 'data/II/']
        folder = self.path
        ann_path = folder + 'label.txt'
        lines = open(ann_path, 'r').readlines()
        # print((lines[0].strip()))
        for line in lines:
            img, gtShape, bndBox = trainSet.read(line, folder)
            scale = 2
            cropB, img = trainSet.cropRegion(bndBox, scale, img)
            gtShape = np.subtract(gtShape,
                                  (cropB[0], cropB[1]))
            # Get the bndBox.
            bndBox = trainSet.getBBoxByPts(gtShape)

            trainSet.add(img, gtShape, bndBox)


        # Generate the meanShape
        trainSet.genTrainData(self.augNum)
        return trainSet

    def printParas(self):
        print('\tDataset     = %s' % (self.path))
        print('\tAugment Num = %d' % (self.augNum))

if __name__ == '__main__':

    config = {
        'name': "face",

        # Different dataset using different reading method
        'dataset': "I",
        'version': "1.0",
        'stageNum': 4,

        'regressorPara':
            {
                'name': 'lbf_reg',
                'para':
                    {
                        'maxTreeNums': [100],
                        'treeDepths': [4],
                        'feaNums': [1000, 750, 500, 375, 250],
                        'radiuses': [0.4, 0.3, 0.2, 0.15, 0.12],
                        # Following para is used to quantize the feature
                        'binNums': [511],
                        'feaRanges': [[-255, 255]],
                    }
            },

        'dataPara':
            {
                'path': "./data/I/",

                # augNum < 1 means don't do augmenting
                'augNum': 1
            }
    }

    dataloder = DataWrapper(config['dataPara'])
    trainSet = dataloder.read()
    cv2.imshow('img',trainSet.imgDatas[100])
    key = cv2.waitKey()
    print(len(trainSet.imgDatas))
