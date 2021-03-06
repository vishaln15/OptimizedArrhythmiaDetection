# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import numpy as np
import os
import sys
import edgeml_pytorch.utils as utils
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class BonsaiTrainer:

    def __init__(self, bonsaiObj, lW, lT, lV, lZ, sW, sT, sV, sZ,
                 learningRate, useMCHLoss=False, outFile=None, device=None):
        '''
        bonsaiObj - Initialised Bonsai Object and Graph
        lW, lT, lV and lZ are regularisers to Bonsai Params
        sW, sT, sV and sZ are sparsity factors to Bonsai Params
        learningRate - learningRate for optimizer
        useMCHLoss - For choice between HingeLoss vs CrossEntropy
        useMCHLoss - True - MultiClass - multiClassHingeLoss
        useMCHLoss - False - MultiClass - crossEntropyLoss
        '''

        self.bonsaiObj = bonsaiObj

        self.lW = lW
        self.lV = lV
        self.lT = lT
        self.lZ = lZ

        self.sW = sW
        self.sV = sV
        self.sT = sT
        self.sZ = sZ

        if device is None:
            self.device = "cpu"
        else:
            self.device = device

        self.useMCHLoss = useMCHLoss

        if outFile is not None:
            print("Outfile : ", outFile)
            self.outFile = open(outFile, 'w')
        else:
            self.outFile = sys.stdout

        self.learningRate = learningRate

        self.assertInit()

        self.optimizer = self.optimizer()

        if self.sW > 0.99 and self.sV > 0.99 and self.sZ > 0.99 and self.sT > 0.99:
            self.isDenseTraining = True
        else:
            self.isDenseTraining = False

    def loss(self, logits, labels):
        '''
        Loss function for given Bonsai Obj
        '''
        regLoss = 0.5 * (self.lZ * (torch.norm(self.bonsaiObj.Z)**2) +
                         self.lW * (torch.norm(self.bonsaiObj.W)**2) +
                         self.lV * (torch.norm(self.bonsaiObj.V)**2) +
                         self.lT * (torch.norm(self.bonsaiObj.T))**2)

        if (self.bonsaiObj.numClasses > 2):
            if self.useMCHLoss is True:
                marginLoss = utils.multiClassHingeLoss(logits, labels)
            else:
                marginLoss = utils.crossEntropyLoss(logits, labels)
            loss = marginLoss + regLoss
        else:
            marginLoss = utils.binaryHingeLoss(logits, labels)
            loss = marginLoss + regLoss

        return loss, marginLoss, regLoss

    def optimizer(self):
        '''
        Optimizer for Bonsai Params
        '''
        optimizer = torch.optim.Adam(
            self.bonsaiObj.parameters(), lr=self.learningRate)

        return optimizer

    def accuracy(self, logits, labels):
        '''
        Accuracy fucntion to evaluate accuracy when needed
        '''
        if (self.bonsaiObj.numClasses > 2):
            correctPredictions = (logits.argmax(dim=1) == labels.argmax(dim=1))
            accuracy = torch.mean(correctPredictions.float())
        else:
            pred = (torch.cat((torch.zeros(logits.shape),
                               logits), 1)).argmax(dim=1)
            accuracy = torch.mean((labels.view(-1).long() == pred).float())

        return accuracy
    
    def classificationReport(self, logits, labels):
        pred = (torch.cat((torch.zeros(logits.shape),
                        logits), 1)).argmax(dim=1)
        return classification_report(labels, pred, output_dict=True)
    
    def confusion_matrix_FAR(self, logits, labels):
        pred = (torch.cat((torch.zeros(logits.shape),
                        logits), 1)).argmax(dim=1)
        CM = confusion_matrix(labels, pred)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        FAR = FP/(FP+TN)
        
        return CM, FAR
    
    def f1(self, logits, labels):
        '''
        f1 score function to evaluate f1 when needed
        '''
#         print("logits:", logits, logits.shape)
#         print("labels:", labels, labels.shape)
        if (self.bonsaiObj.numClasses > 2): # doesnt work for multi-class
            correct = (logits.argmax(dim=1) == labels.argmax(dim=1))
            pred = torch.zeros(logits.shape)
            pred[logits.argmax(dim=1),:] = 1
        else:
            pred = (torch.cat((torch.zeros(logits.shape),
                           logits), 1)).argmax(dim=1)
#         print("pred:", pred, pred.shape)
        f1score = f1_score(labels, pred)

        return f1score

    def runHardThrsd(self):
        '''
        Function to run the IHT routine on Bonsai Obj
        '''
        currW = self.bonsaiObj.W.data
        currV = self.bonsaiObj.V.data
        currZ = self.bonsaiObj.Z.data
        currT = self.bonsaiObj.T.data

        __thrsdW = utils.hardThreshold(currW.cpu(), self.sW)
        __thrsdV = utils.hardThreshold(currV.cpu(), self.sV)
        __thrsdZ = utils.hardThreshold(currZ.cpu(), self.sZ)
        __thrsdT = utils.hardThreshold(currT.cpu(), self.sT)

        self.bonsaiObj.W.data = torch.FloatTensor(
            __thrsdW).to(self.device)
        self.bonsaiObj.V.data = torch.FloatTensor(
            __thrsdV).to(self.device)
        self.bonsaiObj.Z.data = torch.FloatTensor(
            __thrsdZ).to(self.device)
        self.bonsaiObj.T.data = torch.FloatTensor(
            __thrsdT).to(self.device)

        self.__thrsdW = torch.FloatTensor(
            __thrsdW.detach().clone()).to(self.device)
        self.__thrsdV = torch.FloatTensor(
            __thrsdV.detach().clone()).to(self.device)
        self.__thrsdZ = torch.FloatTensor(
            __thrsdZ.detach().clone()).to(self.device)
        self.__thrsdT = torch.FloatTensor(
            __thrsdT.detach().clone()).to(self.device)

    def runSparseTraining(self):
        '''
        Function to run the Sparse Retraining routine on Bonsai Obj
        '''
        currW = self.bonsaiObj.W.data
        currV = self.bonsaiObj.V.data
        currZ = self.bonsaiObj.Z.data
        currT = self.bonsaiObj.T.data

        newW = utils.copySupport(self.__thrsdW, currW)
        newV = utils.copySupport(self.__thrsdV, currV)
        newZ = utils.copySupport(self.__thrsdZ, currZ)
        newT = utils.copySupport(self.__thrsdT, currT)

        self.bonsaiObj.W.data = newW
        self.bonsaiObj.V.data = newV
        self.bonsaiObj.Z.data = newZ
        self.bonsaiObj.T.data = newT

    def assertInit(self):
        err = "sparsity must be between 0 and 1"
        assert self.sW >= 0 and self.sW <= 1, "W " + err
        assert self.sV >= 0 and self.sV <= 1, "V " + err
        assert self.sZ >= 0 and self.sZ <= 1, "Z " + err
        assert self.sT >= 0 and self.sT <= 1, "T " + err

    def saveParams(self, currDir):
        '''
        Function to save Parameter matrices into a given folder
        '''
        paramDir = currDir + '/'
        np.save(paramDir + "W.npy", self.bonsaiObj.W.data.cpu())
        np.save(paramDir + "V.npy", self.bonsaiObj.V.data.cpu())
        np.save(paramDir + "T.npy", self.bonsaiObj.T.data.cpu())
        np.save(paramDir + "Z.npy", self.bonsaiObj.Z.data.cpu())
        hyperParamDict = {'dataDim': self.bonsaiObj.dataDimension,
                          'projDim': self.bonsaiObj.projectionDimension,
                          'numClasses': self.bonsaiObj.numClasses,
                          'depth': self.bonsaiObj.treeDepth,
                          'sigma': self.bonsaiObj.sigma}
        hyperParamFile = paramDir + 'hyperParam.npy'
        np.save(hyperParamFile, hyperParamDict)

    def saveParamsForSeeDot(self, currDir):
        '''
        Function to save Parameter matrices into a given folder for SeeDot compiler
        '''
        seeDotDir = currDir + '/SeeDot/'

        if os.path.isdir(seeDotDir) is False:
            try:
                os.mkdir(seeDotDir)
            except OSError:
                print("Creation of the directory %s failed" %
                      seeDotDir)

        np.savetxt(seeDotDir + "W",
                   utils.restructreMatrixBonsaiSeeDot(self.bonsaiObj.W.data.cpu(),
                                                      self.bonsaiObj.numClasses,
                                                      self.bonsaiObj.totalNodes),
                   delimiter="\t")
        np.savetxt(seeDotDir + "V",
                   utils.restructreMatrixBonsaiSeeDot(self.bonsaiObj.V.data.cpu(),
                                                      self.bonsaiObj.numClasses,
                                                      self.bonsaiObj.totalNodes),
                   delimiter="\t")
        np.savetxt(seeDotDir + "T", self.bonsaiObj.T.data.cpu(), delimiter="\t")
        np.savetxt(seeDotDir + "Z", self.bonsaiObj.Z.data.cpu(), delimiter="\t")
        np.savetxt(seeDotDir + "Sigma",
                   np.array([self.bonsaiObj.sigma]), delimiter="\t")

    def loadModel(self, currDir):
        '''
        Load the Saved model and load it to the model using constructor
        Returns two dict one for params and other for hyperParams
        '''
        paramDir = currDir + '/'
        paramDict = {}
        paramDict['W'] = np.load(paramDir + "W.npy")
        paramDict['V'] = np.load(paramDir + "V.npy")
        paramDict['T'] = np.load(paramDir + "T.npy")
        paramDict['Z'] = np.load(paramDir + "Z.npy")
        hyperParamDict = np.load(paramDir + "hyperParam.npy").item()
        return paramDict, hyperParamDict

    # Function to get aimed model size
    def getModelSize(self):
        '''
        Function to get aimed model size
        '''
        nnzZ, sizeZ, sparseZ = utils.estimateNNZ(self.bonsaiObj.Z, self.sZ)
        nnzW, sizeW, sparseW = utils.estimateNNZ(self.bonsaiObj.W, self.sW)
        nnzV, sizeV, sparseV = utils.estimateNNZ(self.bonsaiObj.V, self.sV)
        nnzT, sizeT, sparseT = utils.estimateNNZ(self.bonsaiObj.T, self.sT)

        totalnnZ = (nnzZ + nnzT + nnzV + nnzW)
        totalSize = (sizeZ + sizeW + sizeV + sizeT)
        hasSparse = (sparseW or sparseV or sparseT or sparseZ)
        return totalnnZ, totalSize, hasSparse

    def train(self, batchSize, totalEpochs,
              Xtrain, Xtest, Ytrain, Ytest, dataDir, currDir):
        '''
        The Dense - IHT - Sparse Retrain Routine for Bonsai Training
        '''
        resultFile = open(dataDir + '/PyTorchBonsaiResults.txt', 'a+')
        numIters = Xtrain.shape[0] / batchSize

        totalBatches = numIters * totalEpochs

        self.sigmaI = 1

        counter = 0
        if self.bonsaiObj.numClasses > 2:
            trimlevel = 15
        else:
            trimlevel = 5
        ihtDone = 0

        maxTestAcc = -10000
        finalF1 = -10000
        finalTrainLoss = -10000
        finalTrainAcc = -10000
        finalClassificationReport = None
        finalFAR = -10000
        finalCM = None
        if self.isDenseTraining is True:
            ihtDone = 1
            self.sigmaI = 1
            itersInPhase = 0

        header = '*' * 20
        for i in range(totalEpochs):
            print("\nEpoch Number: " + str(i), file=self.outFile)

            '''
            trainAcc -> For Classification, it is 'Accuracy'.
            '''
            trainAcc = 0.0
            trainLoss = 0.0

            numIters = int(numIters)
            for j in range(numIters):

                if counter == 0:
                    msg = " Dense Training Phase Started "
                    print("\n%s%s%s\n" %
                          (header, msg, header), file=self.outFile)

                # Updating the indicator sigma
                if ((counter == 0) or (counter == int(totalBatches / 3.0)) or
                        (counter == int(2 * totalBatches / 3.0))) and (self.isDenseTraining is False):
                    self.sigmaI = 1
                    itersInPhase = 0

                elif (itersInPhase % 100 == 0):
                    indices = np.random.choice(Xtrain.shape[0], 100)
                    batchX = Xtrain[indices, :]
                    batchY = Ytrain[indices, :]
                    batchY = np.reshape(
                        batchY, [-1, self.bonsaiObj.numClasses])

                    Teval = self.bonsaiObj.T.data
                    Xcapeval = (torch.matmul(self.bonsaiObj.Z, torch.t(
                        batchX.to(self.device))) / self.bonsaiObj.projectionDimension).data

                    sum_tr = 0.0
                    for k in range(0, self.bonsaiObj.internalNodes):
                        sum_tr += (
                            np.sum(np.abs(np.dot(Teval[k].cpu(), Xcapeval.cpu()))))

                    if(self.bonsaiObj.internalNodes > 0):
                        sum_tr /= (100 * self.bonsaiObj.internalNodes)
                        sum_tr = 0.1 / sum_tr
                    else:
                        sum_tr = 0.1
                    sum_tr = min(
                        1000, sum_tr * (2**(float(itersInPhase) /
                                            (float(totalBatches) / 30.0))))

                    self.sigmaI = sum_tr

                itersInPhase += 1
                batchX = Xtrain[j * batchSize:(j + 1) * batchSize]
                batchY = Ytrain[j * batchSize:(j + 1) * batchSize]
                batchY = np.reshape(
                    batchY, [-1, self.bonsaiObj.numClasses])

                self.optimizer.zero_grad()
                
                logits, _ = self.bonsaiObj(batchX.to(self.device), self.sigmaI)
                batchLoss, _, _ = self.loss(logits, batchY.to(self.device))
                batchAcc = self.accuracy(logits, batchY.to(self.device))

                batchLoss.backward()
                self.optimizer.step()

                # Classification.

                trainAcc += batchAcc.item()
                trainLoss += batchLoss.item()

                # Training routine involving IHT and sparse retraining
                if (counter >= int(totalBatches / 3.0) and
                    (counter < int(2 * totalBatches / 3.0)) and
                    counter % trimlevel == 0 and
                        self.isDenseTraining is False):
                    self.runHardThrsd()
                    if ihtDone == 0:
                        msg = " IHT Phase Started "
                        print("\n%s%s%s\n" %
                              (header, msg, header), file=self.outFile)
                    ihtDone = 1
                elif ((ihtDone == 1 and counter >= int(totalBatches / 3.0) and
                       (counter < int(2 * totalBatches / 3.0)) and
                       counter % trimlevel != 0 and
                       self.isDenseTraining is False) or
                        (counter >= int(2 * totalBatches / 3.0) and
                            self.isDenseTraining is False)):
                    self.runSparseTraining()
                    if counter == int(2 * totalBatches / 3.0):
                        msg = " Sparse Retraining Phase Started "
                        print("\n%s%s%s\n" %
                              (header, msg, header), file=self.outFile)
                counter += 1

            print("\nClassification Train Loss: " + str(trainLoss / numIters) +
                  "\nTraining accuracy (Classification): " +
                  str(trainAcc / numIters),
                  file=self.outFile)
            
            #####################################
            finalTrainAcc = trainAcc / numIters
            finalTrainLoss = trainLoss / numIters
            

            oldSigmaI = self.sigmaI
            self.sigmaI = 1e9
            
            ###################HERE####################################
            
            logits, _ = self.bonsaiObj(Xtest.to(self.device), self.sigmaI)
            testLoss, marginLoss, regLoss = self.loss(
                logits, Ytest.to(self.device))
            testAcc = self.accuracy(logits, Ytest.to(self.device)).item()
            testf1 = self.f1(logits, Ytest.to(self.device))
            testclass = self.classificationReport(logits, Ytest.to(self.device))
            CM, FAR = self.confusion_matrix_FAR(logits, Ytest.to(self.device))

            if ihtDone == 0:
                maxTestAcc = -10000
                maxTestAccEpoch = i
            else:
                if maxTestAcc <= testAcc:
                    maxTestAccEpoch = i
                    maxTestAcc = testAcc
                    self.saveParams(currDir)
                    self.saveParamsForSeeDot(currDir)

            print("Test accuracy %g" % testAcc, file=self.outFile)
            print("Test F1 ", testf1, file=self.outFile)
            print("Test False Alarm Rate ", FAR, file=self.outFile)
            print("Confusion Matrix \n", CM, file=self.outFile)
            print("Classification Report \n", testclass, file=self.outFile)

            #####################################
            testAcc = testAcc
            maxTestAcc = maxTestAcc
            
            finalF1 = testf1
            finalClassificationReport = testclass
            finalFAR = FAR
            finalCM = CM

            print("MarginLoss + RegLoss: " + str(marginLoss.item()) + " + " +
                  str(regLoss.item()) + " = " + str(testLoss.item()) + "\n",
                  file=self.outFile)
            self.outFile.flush()

            self.sigmaI = oldSigmaI

        # sigmaI has to be set to infinity to ensure
        # only a single path is used in inference
        self.sigmaI = 1e9
        print("\nNon-Zero : " + str(self.getModelSize()[0]) + " Model Size: " +
              str(float(self.getModelSize()[1]) / 1024.0) + " KB hasSparse: " +
              str(self.getModelSize()[2]) + "\n", file=self.outFile)

        print("For Classification, Maximum Test accuracy at compressed" +
              " model size(including early stopping): " +
              str(maxTestAcc) + " at Epoch: " +
              str(maxTestAccEpoch + 1) + "\nFinal Test" +
              " Accuracy: " + str(testAcc), file=self.outFile)

        resultFile.write("MaxTestAcc: " + str(maxTestAcc) +
                         " at Epoch(totalEpochs): " +
                         str(maxTestAccEpoch + 1) +
                         "(" + str(totalEpochs) + ")" + " ModelSize: " +
                         str(float(self.getModelSize()[1]) / 1024.0) +
                         " KB hasSparse: " + str(self.getModelSize()[2]) +
                         " Param Directory: " +
                         str(os.path.abspath(currDir)) + "\n")
        
        ##############################################################
        finalModelSize = float(self.getModelSize()[1]) / 1024.0
        
        print("The Model Directory: " + currDir + "\n")

        resultFile.close()
        self.outFile.flush()

        if self.outFile is not sys.stdout:
            self.outFile.close()
            
        finalClassificationReport['train loss'] = finalTrainLoss
        finalClassificationReport['train acc'] = finalTrainAcc
        finalClassificationReport['test f1'] = finalF1
        finalClassificationReport['model size'] = finalModelSize
        finalClassificationReport['test far'] = finalFAR
        return(finalClassificationReport, finalCM)
            
        
