import math
import numpy as np
import random
from affine import Affine


class RandForest(object):
    def __init__(self,
                 treeDepth=None,
                 treeNum=None,
                 feaNum=None,
                 radius=None,
                 binNum=None,
                 feaRange=None):

        self.treeDepth = treeDepth
        self.treeNum = treeNum
        self.feaNum = feaNum
        self.radius = radius
        self.binNum = binNum
        self.feaRange = feaRange
        self.trees = []

    def train(self, train_set, pointIdx):
        sampleNum = train_set.initShapes.shape[0]

        for n in range(self.treeNum):
            # Get train example indexs by bootstrape
            sampleIdxs = self.bootStrape(sampleNum)

            # Construct tree to train
            tree = RegTree(depth=self.treeDepth,
                           radius=self.radius,
                           feaNum=self.feaNum,
                           binNum=self.binNum,
                           feaRange=self.feaRange)
            tree.train(train_set, pointIdx, sampleIdxs)
            self.trees.append(tree)

    def bootStrape(self, sampleNum):
        """
        Todo : Try different bootstrape method
        """
        overlap = 0.4
        subNum = int(sampleNum/((1-overlap)*self.treeNum))
        if not hasattr(self, 'treeIdx'):
            self.treeIdx = 0
        beg = self.treeIdx*subNum*(1-overlap)
        beg = max(0, int(beg))
        end = min(beg+subNum, sampleNum-1)
        self.treeIdx += 1
        return np.array(range(beg, end+1))


class RegTree(object):
    def __init__(self,
                 depth=None,
                 radius=None,
                 feaNum=None,
                 binNum=None,
                 feaRange=None):
        # paras
        self.depth = depth
        self.radius = radius
        self.feaNum = feaNum
        self.binNum = binNum
        self.feaRange = feaRange

        # tree
        self.leafNum = 0
        self.tree = None

    def genBinaryFea(self, imgData, bndBox, affineT, point):
        tree = self.tree
        imgH, imgW = imgData.shape
        w, h = bndBox[2:4]
        point_a = np.zeros(2, dtype=point.dtype)
        point_b = np.zeros(2, dtype=point.dtype)
        while 'leafIdx' not in tree:
            feaType = tree["feaType"]
            feaRange = tree["feaRange"]
            th = tree["threshold"]

            angle_cos = np.cos(feaType[[1, 3]])
            angle_sin = np.sin(feaType[[1, 3]])
            ms_x_ratio = angle_cos*feaType[[0, 2]]
            ms_y_ratio = angle_sin*feaType[[0, 2]]

            point_a[0] = ms_x_ratio[0]*w
            point_a[1] = ms_y_ratio[0]*h
            point_b[0] = ms_x_ratio[1]*w
            point_b[1] = ms_y_ratio[1]*h

            # convert meanshape coord into real coord
            point_a = Affine.transPntForward(point_a,
                                             affineT)
            point_b = Affine.transPntForward(point_b,
                                             affineT)
            point_a = point_a + point
            point_b = point_b + point

            # TODO use other interpolations
            point_a = np.around(point_a)
            point_b = np.around(point_b)

            # Check with the image size
            point_a[point_a < 0] = 0
            point_b[point_b < 0] = 0
            if point_a[0] > imgW-1:
                point_a[0] = imgW-1
            if point_a[1] > imgH-1:
                point_a[1] = imgH-1
            if point_b[0] > imgW-1:
                point_b[0] = imgW-1
            if point_b[1] > imgH-1:
                point_b[1] = imgH-1

            # print('point_a[1]:{}\n point_a[0]:{}\n point_b[1]:{}\n point_b[0]:{}\n'.format(point_a[1],
            #                                                                                point_a[0],
            #                                                                                point_b[1],
            #                                                                                point_b[0]))
            # print('int(point_a[1]):{}\n int(point_a[0]):{}\n int(point_b[1]):{}\n int(point_b[0]):{}\n'.format(int(point_a[1]),
            #                                                                                                    int(point_a[0]),
            #                                                                                                    int(point_b[1]),
            #                                                                                                    int(point_b[0])))
            # print('point_b dtype: {}'.format(point_b.dtype))
            # Construct the idx list for get the elements
            fea = np.subtract(imgData[int(point_a[1]),
                                      int(point_a[0])],
                              imgData[int(point_b[1]),
                                      int(point_b[0])],
                              dtype=np.float32)

            # get the diff
            fea = (fea-feaRange[0])/feaRange[2]
            if fea <= th:
                tree = tree["left"]
            else:
                tree = tree["right"]

        leafIdx = tree["leafIdx"]
        return leafIdx, self.leafNum

    def train(self, train_set, pointIdx, sampleIdxs):
        self.tree = self.split(train_set,
                               pointIdx,
                               sampleIdxs)

    def split(self, train_set, pointIdx, sampleIdxs):
        tree = {}
        if self.depth < 0 or len(sampleIdxs) < 2:
            tree["leafIdx"] = self.leafNum
            self.leafNum = self.leafNum+1
            return tree

        # Get the current residuals
        errs = train_set.residuals[sampleIdxs, pointIdx]

        # Generate feature types
        feaTypes = self.genFeaType(self.feaNum)

        # Extract the pixel difference feature
        pdFeas = self.genFea(train_set, pointIdx,
                             sampleIdxs, feaTypes)

        # Normalize the feature
        minFeas, maxFeas, feaSteps = self.normalize(pdFeas)

        # Find the best feature and threshold
        bestIdx, th = self.findBestSplit(pdFeas, errs)

        # split left and right leaf recurrently
        lIdx = pdFeas[:, bestIdx] <= th
        rIdx = pdFeas[:, bestIdx] > th
        lSamples = sampleIdxs[lIdx]
        rSamples = sampleIdxs[rIdx]
        self.depth = self.depth - 1
        tree["feaType"] = feaTypes[bestIdx]
        tree["feaRange"] = (minFeas[bestIdx],
                            maxFeas[bestIdx],
                            feaSteps[bestIdx])
        tree["threshold"] = th
        tree["left"] = self.split(train_set,
                                  pointIdx,
                                  lSamples)
        tree["right"] = self.split(train_set,
                                   pointIdx,
                                   rSamples)
        return tree

    def findBestSplit(self, feas, errs):
        sampNum, feaNum = feas.shape
        sortedFeas = np.sort(feas, axis=0)
        lossAndTh = np.zeros((feaNum, 2))

        for idxFea in range(feaNum):
            # Randomly split on each feature
            # TODO choose the best split
            ind = int(sampNum*(0.5 + 0.9*(random.random()-0.5)))
            th = sortedFeas[ind, idxFea]
            lIdx = feas[:, idxFea] <= th
            rIdx = feas[:, idxFea] > th

            lErrs = errs[lIdx]
            rErrs = errs[rIdx]
            lNum = lErrs.shape[0]
            rNum = rErrs.shape[0]
            if lNum < 2:
                lVar = 0
            else:
                lVar = np.sum(np.mean(np.power(lErrs, 2),
                                      axis=0) -
                              np.power(np.mean(lErrs,
                                               axis=0), 2))
            if rNum < 2:
                rVar = 0
            else:
                rVar = np.sum(np.mean(np.power(rErrs, 2),
                                      axis=0) -
                              np.power(np.mean(rErrs,
                                               axis=0), 2))
            lossAndTh[idxFea] = (lNum*lVar + rNum*rVar, th)
        bestFeaIdx = lossAndTh[:, 0].argmin()
        return bestFeaIdx, lossAndTh[bestFeaIdx, 1]

    def genFea(self, train_set, pointIdx, sampleIdxs, feaTypes):
        sampleNum = len(sampleIdxs)
        feaNum = feaTypes.shape[0]
        pdFea = np.zeros((sampleNum, feaNum),
                         dtype=np.float32)

        coord_a = np.zeros((feaNum, 2))
        coord_b = np.zeros((feaNum, 2))
        angle_cos = np.cos(feaTypes[:, [1, 3]])
        angle_sin = np.sin(feaTypes[:, [1, 3]])

        ms_x_ratio = angle_cos*feaTypes[:, [0, 2]]
        ms_y_ratio = angle_sin*feaTypes[:, [0, 2]]

        augNum = train_set.augNum
        for i, idx in enumerate(sampleIdxs):
            T = train_set.ms2reals[idx]
            bndBox = train_set.bndBoxs[idx]
            imgData = train_set.imgDatas[int(idx/augNum)]
            initShape = train_set.initShapes[idx]

            # numpy image: H x W x C
            imgH, imgW = imgData.shape
            # print('imgH:{}, imgW:{}'.format(imgH, imgW))
            w, h = bndBox[2:4]
            coord_a[:, 0] = ms_x_ratio[:, 0]*w
            coord_a[:, 1] = ms_y_ratio[:, 0]*h
            coord_b[:, 0] = ms_x_ratio[:, 1]*w
            coord_b[:, 1] = ms_y_ratio[:, 1]*h

            # convert meanshape coord into real coord
            coord_a = Affine.transPntsForwardWithSameT(coord_a, T)
            coord_b = Affine.transPntsForwardWithSameT(coord_b, T)
            coord_a = coord_a+initShape[pointIdx]
            coord_b = coord_b+initShape[pointIdx]

            # TODO use other interpolations
            coord_a = np.floor(coord_a)
            coord_b = np.floor(coord_b)
            # print('coord_a shape: {}'.format(coord_a.shape))

            # Check with the image size
            coord_a[coord_a < 0] = 0

            # print('coord_a[:,0]>imgW-1: {}'.format(coord_a[:,0]>imgW-1))
            # print('coord_a[:,0]: {}'.format(coord_a[:,0]))
            coord_a[coord_a[:, 0] > imgW-1, 0] = imgW-1
            coord_a[coord_a[:, 1] > imgH-1, 1] = imgH-1
            coord_b[coord_b < 0] = 0
            coord_b[coord_b[:, 0] > imgW-1, 0] = imgW-1
            coord_b[coord_b[:, 1] > imgH-1, 1] = imgH-1

            # Construct the idx list for get the elements
            idx_a = np.transpose(coord_a).tolist()
            idx_a[0], idx_a[1] = idx_a[1], idx_a[0]
            idx_b = np.transpose(coord_b).tolist()
            # print('idx_a: {}'.format(idx_a))
            idx_b[0], idx_b[1] = idx_b[1], idx_b[0]
            idx_a = np.array(idx_a, dtype=np.int64)
            # print(('idx_aa: {}'.format(idx_a[1])))
            idx_b = np.array(idx_b, dtype=np.int64)


            ### get the diff
            # print('i: {}, ind:{}'.format(i, idx))
            # print('imgData shape', imgData.shape)
            # print('idx_a[0]: {}\n idx_a[1]: {} \n idx_b[0]:{}\n idx_b[1]: {}'.format(idx_a[0], idx_a[1], idx_b[0], idx_b[1]))
            # print('idx_a[0] shape: {}'.format(idx_a[0].shape))
            pdFea[i, :] = np.subtract(imgData[idx_a[0], idx_a[1]],
                                      imgData[idx_b[0], idx_b[1]],
                                      dtype=np.int64)
        return pdFea

    def genFeaType(self, num):
        feaType = np.zeros((num, 4), dtype=np.float32)
        radRange, angRange = 30, 36
        a = np.array(range(0, (radRange+1)*(angRange+1)),
                     dtype=np.float32)
        b = np.array(range(0, (radRange+1)*(angRange+1)),
                     dtype=np.float32)
        random.shuffle(a)
        random.shuffle(b)
        dif_idx = a != b
        a = a[dif_idx]
        b = b[dif_idx]
        a = a[0:num]
        b = b[0:num]

        for i in range(num):
            rad_a = math.floor(a[i]/(angRange+1))
            ang_a = math.floor(a[i] % (angRange+1))
            rad_b = math.floor(b[i]/(angRange+1))
            ang_b = math.floor(b[i] % (angRange+1))
            feaType[i, :] = (rad_a/radRange*self.radius,
                             ang_a/angRange*2*math.pi,
                             rad_b/radRange*self.radius,
                             ang_b/angRange*2*math.pi)
        return feaType

    def normalize(self, feas):
        feaDim = feas.shape[1]
        minFeas = np.empty(feaDim,
                           dtype=np.float32)
        maxFeas = np.empty(feaDim,
                           dtype=np.float32)
        feaSteps = np.empty(feaDim,
                           dtype=np.float32)

        if None != self.feaRange:
            minFeas[:] = self.feaRange[0]
            maxFeas[:] = self.feaRange[1]
        else:
            np.min(feas, axis=0, out=minFeas)
            np.max(feas, axis=0, out=maxFeas)
        feaR = (maxFeas - minFeas + 1)
        feaSteps[:] = feaR/self.binNum
        np.subtract(feas, minFeas, out=feas)
        np.divide(feas, feaSteps, out=feas, dtype=np.float32)
        np.round(feas, out=feas)
        return minFeas, maxFeas, feaSteps
