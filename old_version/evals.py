import numpy as np
import json
import copy

import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.autograd import Variable
import torch.nn.functional as F 

from metrics import metric_computation
from fabulous.color import fg256


def interm_evaluation(args, model, metric, weights_name, loaders, current, cnt):
    
    use_gpu = torch.cuda.is_available()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    encoder = model[0]; regressor = model[1]; ERM_FC = model[2]
    RMSE = metric

    count = cnt
    current_dir, current_time = current[0], current[1]
    
    encoder.train(False)
    regressor.train(False)
    ERM_FC.train(False)

    rmse_v, rmse_a = 0., 0.
    cnt, erm_cnt = 0, 0

    z_list, scores_list, labels_list = [], [], []
    with torch.no_grad():
        for batch_i, data_i in enumerate(loaders['val']):
            elist = []
            
            data, emotions, path = data_i['image'], data_i['va'], data_i['path']
            valence = np.expand_dims(np.asarray(emotions[0]), axis=1)
            arousal = np.expand_dims(np.asarray(emotions[1]), axis=1)
            emotions = torch.from_numpy(np.concatenate([valence, arousal], axis=1)).float()
            path = np.asarray(path)

            if use_gpu:
                inputs, correct_labels = Variable(data.cuda()), Variable(emotions.cuda())
            else:
                inputs, correct_labels = Variable(data), Variable(emotions)

            if args.model == 'mlpmixer':
                _, latent_variable = encoder(inputs)
                output = ERM_FC(latent_variable)
            else:
                latent_variable = regressor(encoder(inputs))
                output = ERM_FC(latent_variable)

            scores = output
            scores_list.append(scores.detach().cpu().numpy())
            labels_list.append(correct_labels.detach().cpu().numpy())
            RMSE_valence = RMSE(scores[:,0], correct_labels[:,0])**0.5
            RMSE_arousal = RMSE(scores[:,1], correct_labels[:,1])**0.5
            rmse_v += RMSE_valence.item()
            rmse_a += RMSE_arousal.item()
            cnt = cnt + 1

    PCC_v, PCC_a, CCC_v, CCC_a, SAGR_v, SAGR_a, final_rmse_v, final_rmse_a = \
            metric_computation([rmse_v, rmse_a], scores_list, labels_list, cnt)

    # write results to log file
    if count == args.initial_check:
        with open(current_dir+'/log/'+current_time+'.txt', 'a') as f:
            json.dump(args.__dict__, f, indent=2)

    with open(current_dir+'/log/'+current_time+'.txt', 'a') as f:
        f.writelines(['\nItr: \t{},\n PCC: \t{}|\t {},\n CCC: \t{}|\t {},\n SAGR: \t{}|\t {},\n RMSE: \t{}|\t {}\n\n'.format(count, PCC_v[0,1], PCC_a[0,1], CCC_v[0,1], CCC_a[0,1], SAGR_v, SAGR_a, final_rmse_v, final_rmse_a)])
