import sys
import os
import numpy as np
import pickle
from data_load import DataWrapper
from affine import Affine
from regressorWrapper import RegressorWrapper
from shape import Shape


class Cascador(object):
    """
    Cascade regression for landmark
    """

    def __init__(self):
        self.name = None
        # self.version = None
        self.stageNum = None

        self.dataWrapper = None
        self.regWrapper = None
        self.regressors = []
        self.meanShape = None

    def printParas(self):
        print('------------------------------------------')
        print('----------   Configuration    ------------')
        print('Name           = %s' % (self.name))
        # print('Version        = %s' % (self.version))
        print('Stage Num      = %s' % (self.stageNum))
        print('\n-- Data Config --')
        self.dataWrapper.printParas()
        print('\n-- Regressor Config --')
        self.regWrapper.printParas()
        print('---------   End of Configuration   -------')
        print('------------------------------------------\n')

    def config(self, paras):
        self.name = paras['name']
        # self.version = paras['version']
        self.stageNum = paras['stageNum']

        # Construct the regressor wrapper
        regPara = paras['regressorPara']
        self.regWrapper = RegressorWrapper(regPara)

        # Construct the data wrapper
        dataPara = paras['dataPara']
        if 'dataset' in paras:
            dataPara['dataset'] = paras['dataset']
        self.dataWrapper = DataWrapper(dataPara)

    def train(self, save_path):
        # mkdir model folder for train model
        if not os.path.exists('%s/model' % (save_path)):
            os.mkdir('%s/model' % (save_path))

        # read data first

        trainSet = self.dataWrapper.read()
        dataNum = trainSet.initShapes.shape[0]
        self.meanShape = trainSet.meanShape

        print("\tData Number : %d" % (dataNum))
        trainSet.calResiduals()
        sumR = np.mean(np.abs(trainSet.residuals))
        print("\tManhattan Distance in MeanShape : %f\n" % sumR)

        for idx in range(self.stageNum):
            print("\t%drd stage begin ..." % idx)
            # train one stage
            reg = self.regWrapper.getClassInstance(idx)
            reg.train(trainSet)
            self.regressors.append(reg)

            # calculate the residuals
            trainSet.calResiduals()
            sumR = np.mean(np.abs(trainSet.residuals))
            print("\tManhattan Distance in MeanShape : %f" % sumR)

        self.saveModel(save_path)

    def detect(self, img, bndbox, initShape):
        mShape = Shape.shapeNorm2Real(self.meanShape,
                                      bndbox)
        for reg in self.regressors:
            affineT = Affine.fitGeoTrans(mShape,
                                         initShape)
            reg.detect(img, bndbox, initShape, affineT)

    def loadModel(self, model):
        path_obj = open(model, 'r').readline().strip()
        folder = os.path.split(model)[0]
        objFile = open("%s/%s" % (folder, path_obj), 'rb')
        self = pickle.load(objFile)
        objFile.close()
        return self

    def saveModel(self, save_path):
        name = self.name.lower()
        model_path = "%s/model/train.model" % (save_path)
        model = open(model_path, 'w')
        model.write("%s.pyobj" % (name))
        obj_path = "%s/model/%s.pyobj" % (save_path, name)

        objFile = open(obj_path, 'wb')
        pickle.dump(self, objFile)
        objFile.close()
        model.close()


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
                'augNum': 0
            }
    }
    cascade = Cascador()
    cascade.config(config)
    save_path = './'
    cascade.train(save_path)

