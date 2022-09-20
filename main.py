from __future__ import print_function, division
import os

import argparse
app = argparse.ArgumentParser()
app.add_argument("--folder_name", type=str, default='', help='Checkpoint saving folder.')
app.add_argument("--gpus", type=int, default=3, help='Which GPU do you want for training.')
app.add_argument("--model", type=str, default='alexnet', help='alexnet/resnet18/Mlp-Mixer.')
app.add_argument("--relevance_weighting", type=int, default=1)
app.add_argument("--sinkhorn", type=bool, default=True)
app.add_argument("--matching_method", type=str, choices=['Sinkhorn', 'Cross-attention'], default='Sinkhorn')
app.add_argument("--weight_sampling", type=str, choices=['gumbel-softmax', 'max-filling', 'none'], default='gumbel-softmax')
app.add_argument("--online_tracker", type=int, default=1, help='On(1) and Off(0).')
app.add_argument("--dataset", type=str, choices=['AFEW-VA', 'Aff-wild', 'Aff-wild2'], default='AFEW-VA')

app.add_argument("--optimizer", type=str, choices=['adam', 'adamw', 'sgd'], default='adam', help='Which optimizer to use.')
app.add_argument("--lr", type=float, default=4e-5, help='Learning rate (Default).')
app.add_argument("--seed", type=int, default=1, help='Seed against training randomness.')
app.add_argument("--initial_check", type=int, default=10, help='Save frequency')
app.add_argument("--print_check", type=int, default=250, help='Save frequency')
app.add_argument("--warmup_coef1", type=float, default=10, help='Initial warmup phase.')
app.add_argument("--warmup_coef2", type=float, default=200, help='Real warmup phase.')
app.add_argument("--isa_parametrization", type=int, default=0)
app.add_argument("--epsilon", type=float, default=1e-6, help='Offset value.')

app.add_argument("--no_domain", type=int, default=5, help='Number of domain (of person).')
app.add_argument("--topk", type=int, default=40, help='Minimum length of BoD.')
app.add_argument("--cN", type=int, default=350, help='Dimension size of latent variable.')
app.add_argument("--K", type=int, default=3, help='Number of cluster means.')
app.add_argument("--tr_BS", type=int, default=512, help='Batch size for training.')
app.add_argument("--te_BS", type=int, default=512, help='Batch size for testing.')
app.add_argument("--reg_coef", type=float, default=1e-3, help='Regularization factor of ERM loss.')

app.add_argument("--latent_dim", type=int, default=64, help='Latent dimension of LSTM.')
app.add_argument("--erm_input_dim", type=int, default=64, help='Input dimension of ERM_FC.')
app.add_argument("--erm_output_dim", type=int, default=2, help='Output dimension of ERM_FC.')
app.add_argument("--hidden_dim", type=int, default=8, help='Size of latent feat of Stn/Rsc_NN.')
app.add_argument("--lstm_hidden_dim", type=int, default=512, help='Hidden dimension of LSTM.')
app.add_argument("--lstm_layer_dim", type=int, default=2, help='Depth of LSTM layer.')
app.add_argument("--lstm_output_dim", type=int, default=64, help='Ouput dimension of LSTM.')
app.add_argument("--mode", type=str, default='all', help='LSTM mode (all or last).')
app.add_argument("--num_epochs", type=int, default=100, help='Number of dataset iterations.')
args = app.parse_args()


import pandas as pd
import random
import numpy as np
from numpy.random import default_rng
import time
from time import gmtime, strftime
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.models as models
import torchvision.datasets as datasets

from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from fabulous.color import fg256
import wandb
import pretrainedmodels

from utils import LENGTH_CHECK, pcc_ccc_loss, gumbel_softmax_sample, Sinkhorn_Knopp, Cross_Attention
from models import encoder_AL, encoder_R18, regressor_AL, regressor_R18, regressor_R50, load_ERM_FC
from attention_module import load_Cross_Attention
from evals import interm_evaluation
from dataset_utils import FaceDataset
from big_model import MlpMixer, CONFIGS


def model_training(args, model, optimizer, scheduler, loaders, current):

    if args.online_tracker:
        wandb.init(project="ELIM_NeurIPS2022")

    cnt = 0
    current_dir, current_time = current[0], current[1]
    rng = default_rng()

    encoder = model[0]; regressor = model[1]; ERM_FC = model[2]; CHA = model[3]
    enc_opt = optimizer[0]; reg_opt = optimizer[1]; ermfc_opt = optimizer[2]; cha_opt = optimizer[3]

    MSE = nn.MSELoss()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    layer_norm = nn.LayerNorm(args.latent_dim, elementwise_affine=False).cuda()

    for epoch in range(args.num_epochs):
        print('\nepoch ' + str(epoch) + '/' + str(args.num_epochs-1))

        epoch_iterator = tqdm(loaders['train'],
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        for _, batch in enumerate(epoch_iterator):

            for enc_param_group in enc_opt.param_groups:
                aa = enc_param_group['lr']
            for reg_param_group in reg_opt.param_groups:
                bb = reg_param_group['lr']
            
            data, emotions, path = batch['image'], batch['va'], batch['path']
            valence = np.expand_dims(np.asarray(emotions[0]), axis=1)
            arousal = np.expand_dims(np.asarray(emotions[1]), axis=1)
            emotions = torch.from_numpy(np.concatenate([valence, arousal], axis=1)).float()
            path = np.asarray(path)

            path_list = []
            for i in range(len(path)): path_list.append(path[i].split('/')[0])

            ll_sorted = sorted(set(path_list))
            ll_dictionary = dict()
            for k in range(len(ll_sorted)):
                ll_dictionary.update([[ll_sorted[k], [i for i, j in enumerate(path) if j.split('/')[0] == ll_sorted[k]] ]])

            if use_gpu:
                inputs, correct_labels = Variable(data.cuda()), Variable(emotions.cuda())
            else:
                inputs, correct_labels = Variable(data), Variable(emotions)

            # ---------------------
            # Conventional learning
            # ---------------------
            if args.model == 'mlpmixer':
                _, latent_feats = encoder(inputs)
                scores = ERM_FC(latent_feats)
            else:
                latent_feats = regressor(encoder(inputs))
                scores = ERM_FC(latent_feats)

            pcc_loss, ccc_loss, ccc_v, ccc_a = pcc_ccc_loss(correct_labels, scores)

            MSE_v = MSE(scores[:,0], correct_labels[:,0])
            MSE_a = MSE(scores[:,1], correct_labels[:,1])

            enc_opt.zero_grad()
            reg_opt.zero_grad()
            ermfc_opt.zero_grad()
            loss = (MSE_v + MSE_a) + (0.5 * pcc_loss + 0.5 * ccc_loss)
            loss.backward(retain_graph=True)

            enc_opt.step()
            reg_opt.step()
            ermfc_opt.step()

            
            # ----------------------------------------
            # Domain quantification and loss functions
            # ----------------------------------------
            # Warm-up
            if cnt < args.warmup_coef1:
                temperature = 0.
            elif cnt >= args.warmup_coef1 and cnt < args.warmup_coef2:
                temperature = cnt / args.warmup_coef2
            else:
                temperature = 1.

            if temperature:

                enc_opt.zero_grad()
                reg_opt.zero_grad()
                ermfc_opt.zero_grad()
                cha_opt.zero_grad()
                stnnn_opt.zero_grad()
                rscnn_opt.zero_grad()
    
                domain_label = np.arange(len(ll_sorted))
    
                if args.model == 'mlpmixer':
                    _, latent_variable = encoder(inputs)
                else:
                    latent_variable = regressor(encoder(inputs))
    
                # Pre-processing: ID grouping
                try:
                    rnd_sample = rng.choice(domain_label, size=args.no_domain, replace=False)
                except ValueError:
                    rnd_sample = rng.choice(domain_label, size=2, replace=False)
    
                vector_dict = dict(); label_dict = dict()
                labmn_dict = dict(); proto_dict = dict()
                domain_id_list = []
                for abc in range(len(rnd_sample)):
    
                    domain_id_list.append(abc)  # list domain id
    
                    mu       = latent_variable[ ll_dictionary[ll_sorted[rnd_sample[abc]]] ]
                    label_gt = correct_labels[ ll_dictionary[ll_sorted[rnd_sample[abc]]] ]
                    label_mn = torch.FloatTensor([label_gt[i].norm(p=2) for i in range(label_gt.size(0))])
                    label_mn = F.normalize(label_mn.unsqueeze(0), p=1).log()
                    gumbel_sm = gumbel_softmax_sample(label_mn, 0.5)
    
                    vector_dict.update([[abc, mu]]); label_dict.update([[abc, label_gt]])
                    labmn_dict.update([[abc, gumbel_sm]])
    
                # validity check
                if len(domain_id_list) == 0:
                    print(fg256("red", "BoD is EMPTY!!, Skip this iteration ;("))
                    continue
    
    
                # ------------------
                # Align shape of BoP
                # ------------------
                if args.weight_sampling == 'gumbel-softmax':
                    for _, i in enumerate(domain_id_list):
                        _, indices = torch.topk(labmn_dict[i],args.topk//2)
                        vector_dict[i] = vector_dict[i][indices].squeeze(0)
                        label_dict[i] = label_dict[i][indices].squeeze(0)
                elif args.weight_sampling == 'max-filling':
                    user_no_list = [len(vector_dict[i]) for _, i in enumerate(domain_id_list)]
                    max_user_no = np.max(user_no_list)
                    for nnn, i in enumerate(domain_id_list):
                        user_iter = max_user_no - user_no_list[nnn]
                        if user_iter == 0: continue
                        else:
                            vector_dict[i] = torch.cat([vector_dict[i], vector_dict[i][-1].repeat(user_iter,1)], dim=0)
                            label_dict[i]  = torch.cat([label_dict[i], label_dict[i][-1].repeat(user_iter,1)], dim=0)
                elif args.weight_sampling == 'none':
                    pass

                # Quantify domain shift
                vectors = [vector_dict[i] for i in sorted(vector_dict)]  # convert `dict` to `list`
                labels  = [label_dict[i] for i in sorted(label_dict)]
                if args.matching_method == 'Sinkhorn':
                    domain_shift = Sinkhorn_Knopp(args, vectors, cos, sinkhorn=args.sinkhorn)
                elif args.matching_method == 'Cross-attention':
                    domain_shift = Cross_Attention(args, CHA, vectors)
    
                # --------------------
                # Domain-wise ERM loss
                # --------------------
                total_erm_loss, total_reg_loss = 0., 0.
                erm_loss, reg_loss = 0., 0.
                for nnn, usr_idx in enumerate(domain_id_list):
                    aaa = LENGTH_CHECK(vector_dict[usr_idx])
                    if usr_idx == domain_id_list[0]:
                        erm_loss = 0.
                        outputs = ERM_FC(aaa).float()
                        erm_loss = MSE(outputs, label_dict[usr_idx])  #MSE
                    else:
                        upper = aaa - domain_shift[nnn-1][0]
                        lower = upper.pow(2).sum(dim=1) / args.latent_dim
                        outputs = ERM_FC( upper / (torch.sqrt(lower.unsqueeze(1))+args.epsilon) ).float()

                        erm_loss = MSE(outputs, label_dict[usr_idx])  #MSE
                    total_erm_loss += erm_loss
    
                total_loss = args.reg_coef*total_erm_loss
                total_loss = temperature * total_loss
                enc_opt.zero_grad(); reg_opt.zero_grad(); ermfc_opt.zero_grad()
                if args.matching_method == 'Cross-attention':
                    cha_opt.zero_grad()
    
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)  # gradient clipping

                enc_opt.step(); reg_opt.step(); ermfc_opt.step()
                if args.matching_method == 'Cross-attention':
                    cha_opt.step()
            
                if args.online_tracker:
                    wandb.log({
                        "loss": loss.item(),
                        "ERM loss": total_erm_loss.item(),
                        "Enc_lr": aa, "Reg_lr": bb,
                        "epoch": epoch, "ccc_v": ccc_v.item(), "ccc_a": ccc_a.item(),
                        "RMSE (v)": MSE_v.item(), "RMSE (a)": MSE_a.item(),
                    })
            

            if cnt % args.print_check == 0 and cnt > 10 or cnt == args.initial_check:
                encoder_name = '{}/checkpoint/{}/enc_{}_{}.t7'.format(args.dataset, args.folder_name, cnt, epoch)
                regressor_name = '{}/checkpoint/{}/reg_{}_{}.t7'.format(args.dataset, args.folder_name, cnt, epoch)
                ERM_FC_name = '{}/checkpoint/{}/erm_fc_{}_{}.t7'.format(args.dataset, args.folder_name, cnt, epoch)
                torch.save(encoder.state_dict(), encoder_name)
                torch.save(regressor.state_dict(), regressor_name)
                torch.save(ERM_FC.state_dict(), ERM_FC_name)

                # validation step
                encoder.train(False)
                regressor.train(False)
                ERM_FC.train(False)

                interm_evaluation(args, [encoder, regressor, ERM_FC], MSE,
                                  [encoder_name, regressor_name], loaders, current, cnt)

                encoder.train(True)
                regressor.train(True)
                ERM_FC.train(True)

            cnt = cnt + 1

            scheduler[0].step()  # update LR schedule
            scheduler[1].step()
            scheduler[2].step()
            scheduler[3].step()


if __name__ == "__main__":

    import warnings
    warnings.filterwarnings("ignore")

    if not os.path.exists('{}/checkpoint/{}'.format(args.dataset, args.folder_name)):
        os.makedirs('{}/checkpoint/{}'.format(args.dataset, args.folder_name))

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    current_dir  = os.getcwd()
    current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())

    if args.model == 'alexnet':
        with open(current_dir+'/log/'+current_time+'.txt', 'w') as f:
            f.writelines(["Title: {} ELIM AlexNet.\n".format(args.dataset)])
    elif args.model == 'resnet18' or args.model == 'resnet50':
        with open(current_dir+'/log/'+current_time+'.txt', 'w') as f:
            f.writelines(["Title: {} ELIM ResNet.\n".format(args.dataset)])
    else:
        with open(current_dir+'/log/'+current_time+'.txt', 'w') as f:
            f.writelines(["Title: {} ELIM BIG-MODEL.\n".format(args.dataset)])

    #----------------
    # Build DNN model
    #----------------
    if args.model == 'alexnet':
        print(fg256('green', 'Choose AlexNet'))
        encoder    = encoder_AL().to(device)
        regressor  = regressor_AL(args.latent_dim).to(device)
    elif args.model == 'resnet18':
        print(fg256('orange', 'Choose ResNet18'))
        encoder    = nn.DataParallel(encoder_R18()).to(device)
        regressor  = regressor_R18(args.latent_dim).to(device)
    elif args.model == 'resnet50':
        print(fg256('white', 'Choose ResNet50'))
        encoder    = models.resnet50(pretrained=True).cuda()
        encoder    = nn.DataParallel(encoder).to(device)
        regressor  = regressor_R50(args.latent_dim).to(device)
    elif args.model == 'mlpmixer':
        print(fg256('cyan', 'Choose MLP-Mixer'))
        config = CONFIGS['Mixer-B_16']
        encoder = MlpMixer(config, img_size=224, num_classes=2, patch_size=16, latent_dim=args.latent_dim, zero_head=True)
        encoder.load_from(np.load('mlpmixer_checkpoint/INet_1K/Mixer-B_16.npz'))
        encoder    = nn.DataParallel(encoder).to(device)
        regressor  = regressor_R18(args.latent_dim).to(device)

    ERM_FC = load_ERM_FC(args.erm_input_dim, args.erm_output_dim).to(device)
    CHA    = load_Cross_Attention(args.latent_dim).to(device)

    #-------------
    # Load dataset
    #-------------
    training_path = 'training.csv'
    validation_path = 'validation.csv'

    face_dataset = FaceDataset(csv_file=training_path,
                               root_dir='/',
                               transform=transforms.Compose([
                                   transforms.Resize(256), transforms.RandomCrop(size=224),
                                   transforms.ColorJitter(),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                               ]), inFolder=None)

    face_dataset_val = FaceDataset(csv_file=validation_path,
                               root_dir='/',
                               transform=transforms.Compose([
                                   transforms.Resize(256), transforms.CenterCrop(size=224),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                               ]), inFolder=None)
    
    dataloader = DataLoader(face_dataset, batch_size=args.tr_BS, shuffle=True)  #, drop_last=True)
    dataloader_val = DataLoader(face_dataset_val, batch_size=args.te_BS, shuffle=True)  # False
    
    loaders = {'train': dataloader, 'val': dataloader_val}
    
    dataset_size = {'train': len(face_dataset), 'val': len(face_dataset_val)}
    train_size = dataset_size['train']; val_size = dataset_size['val']
    print(fg256("yellow", 'train | val size: {} | {}'.format(train_size, val_size)))


    #------------------
    # Optimizer setting
    #------------------
    enc_opt   = optim.Adam(encoder.parameters(),   lr=args.lr,     betas=(0.5, 0.9))
    reg_opt   = optim.Adam(regressor.parameters(), lr=args.lr,     betas=(0.5, 0.9))
    cha_opt   = optim.Adam(CHA.parameters(),       lr=args.lr,     betas=(0.5, 0.9))
    ermfc_opt = optim.Adam(ERM_FC.parameters(),    lr=0.2*args.lr, betas=(0.5, 0.9))
    
    enc_exp_lr_sc   = lr_scheduler.MultiStepLR(enc_opt,   milestones=[5e3,25e3,45e3,65e3], gamma=0.8)
    reg_exp_lr_sc   = lr_scheduler.MultiStepLR(reg_opt,   milestones=[5e3,25e3,45e3,65e3], gamma=0.8)
    ermfc_exp_lr_sc = lr_scheduler.MultiStepLR(ermfc_opt, milestones=[5e3,25e3,45e3,65e3], gamma=0.8)
    cha_exp_lr_sc   = lr_scheduler.MultiStepLR(cha_opt,   milestones=[5e3,25e3,45e3,65e3], gamma=0.8)
    

    #---------------
    # Start training
    #---------------
    model_training(args, [encoder, regressor, ERM_FC, CHA], [enc_opt, reg_opt, ermfc_opt, cha_opt],
                   [enc_exp_lr_sc, reg_exp_lr_sc, ermfc_exp_lr_sc, cha_exp_lr_sc],
                   loaders, [current_dir, current_time])
