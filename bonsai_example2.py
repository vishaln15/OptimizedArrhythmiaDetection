# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import helpermethods
import numpy as np
import sys
import pandas as pd
from edgeml_pytorch.trainer.bonsaiTrainer2 import BonsaiTrainer
from edgeml_pytorch.graph.bonsai import Bonsai
import torch
from sklearn.model_selection import KFold
import json

def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def main():
    # change cuda:0 to cuda:gpuid for specific allocation
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # Fixing seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Hyper Param pre-processing
    args = helpermethods.getArgs()

    sigma = args.sigma
    depth = args.depth

    projectionDimension = args.proj_dim
    regZ = args.rZ
    regT = args.rT
    regW = args.rW
    regV = args.rV

    totalEpochs = args.epochs

    learningRate = args.learning_rate

    dataDir = args.data_dir
    
    DFPATH = args.df_path

    outFile = args.output_file
    
    k_folds = args.kF
    
    ######################HERE#################################
    
    final_scores = []
    final_scores_with_cm = []
    df = pd.read_csv(DFPATH)
    
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    for train_index, test_index in kfold.split(df):
        train = df.iloc[train_index] 
        test = df.iloc[test_index] 
#         print(train)
#         print(test)

        train = train.to_numpy()
        test = test.to_numpy()
        
        np.save(dataDir + '/train.npy', train)
        np.save(dataDir + '/test.npy', test)

        (dataDimension, numClasses, Xtrain, Ytrain, Xtest, Ytest,
         mean, std) = helpermethods.preProcessData(dataDir)

        sparZ = args.sZ

        if numClasses > 2:
            sparW = 0.2
            sparV = 0.2
            sparT = 0.2
        else:
            sparW = 1
            sparV = 1
            sparT = 1

        if args.sW is not None:
            sparW = args.sW
        if args.sV is not None:
            sparV = args.sV
        if args.sT is not None:
            sparT = args.sT

        if args.batch_size is None:
            batchSize = np.maximum(100, int(np.ceil(np.sqrt(Ytrain.shape[0]))))
        else:
            batchSize = args.batch_size

        useMCHLoss = True

        if numClasses == 2:
            numClasses = 1

        currDir = helpermethods.createTimeStampDir(dataDir)

        helpermethods.dumpCommand(sys.argv, currDir)
        helpermethods.saveMeanStd(mean, std, currDir)

        # numClasses = 1 for binary case
        bonsaiObj = Bonsai(numClasses, dataDimension,
                           projectionDimension, depth, sigma).to(device)

        bonsaiTrainer = BonsaiTrainer(bonsaiObj,
                                      regW, regT, regV, regZ,
                                      sparW, sparT, sparV, sparZ,
                                      learningRate, useMCHLoss, outFile, device)

        fold_scores, CM = bonsaiTrainer.train(batchSize, totalEpochs,
                            torch.from_numpy(Xtrain.astype(np.float32)),
                            torch.from_numpy(Xtest.astype(np.float32)),
                            torch.from_numpy(Ytrain.astype(np.float32)),
                            torch.from_numpy(Ytest.astype(np.float32)),
                            dataDir, currDir)
        
        fold_scores = pd.json_normalize(fold_scores, sep='_')
        fold_scores = fold_scores.to_dict(orient='records')[0]
        fold_scores_with_cm = fold_scores.copy()
        fold_scores_with_cm['CM'] = CM.tolist()
        final_scores_with_cm.append(fold_scores_with_cm)


        print('########################################## FOLD SCORES ############################################')
        print(fold_scores)
        final_scores.append(fold_scores)
    
    print('########################################## FINAL SCORES ############################################')
    avg_score = dict_mean(final_scores)
    print(avg_score)
    
    with open(dataDir + 'Fold Results tf e-300 kF-5 depth-3.txt', 'w') as file:
        file.write(json.dumps(final_scores_with_cm, indent=4))
        
    with open(dataDir + 'Final Results tf e-300 kF-5 depth-3.txt', 'w') as file:
        file.write(json.dumps(avg_score, indent=4))
    
    
    sys.stdout.close()


if __name__ == '__main__':
    main()
