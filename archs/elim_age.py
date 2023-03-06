import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

from . import BasicTask
from common.dataset_utils import FaceDataset_AffectNet
from common.ops import convert_to_ddp
from common.metrics import metric_computation
from common.utils import LENGTH_CHECK, pcc_ccc_loss, gumbel_softmax_sample, Sinkhorn_Knopp, Cross_Attention
from common.big_model import MlpMixer, CONFIGS

import wandb
from fabulous.color import fg256


class ELIM_Age(BasicTask):

    def set_loader(self):
        opt = self.opt

        training_path = opt.data_path+'train/training.csv'
        validation_path = opt.data_path+'val/validation.csv'

        face_dataset = FaceDataset_AffectNet(csv_file=training_path,
                                   root_dir=opt.data_path+'train/',
                                   transform=transforms.Compose([
                                       transforms.Resize(256), transforms.RandomCrop(size=224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                   ]), inFolder=None)
    
        face_dataset_val = FaceDataset_AffectNet(csv_file=validation_path,
                                       root_dir=opt.data_path+'val/',
                                       transform=transforms.Compose([
                                           transforms.Resize(256), transforms.CenterCrop(size=224),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                       ]), inFolder=None)
    
    
        train_loader = DataLoader(face_dataset, batch_size=opt.tr_batch_size, shuffle=True)
        val_loader   = DataLoader(face_dataset_val, batch_size=opt.te_batch_size, shuffle=False)  # False

        self.train_loader = train_loader
        self.val_loader = val_loader


    def set_model(self):
        opt = self.opt

        from common.elim_networks import encoder_AL, encoder_R18, regressor_AL, regressor_R18, load_ERM_FC
        from common.attention_module import load_Cross_Attention
        if opt.model == 'alexnet':
            print(fg256("yellow", '[network] AlexNet loaded.'))
            encoder      = encoder_AL().cuda()
            regressor    = regressor_AL(opt.ermfc_input_dim).cuda()
        elif opt.model == 'resnet18':
            print(fg256("yellow", '[network] ResNet18 loaded.'))
            encoder    = encoder_R18().cuda()
            regressor  = regressor_R18(opt.ermfc_input_dim).cuda()
        elif opt.model == 'mlpmixer':
            print(fg256("yellow", '[network] Mlp-Mixer loaded.'))
            config = CONFIGS['Mixer-B_16']
            encoder = MlpMixer(config, img_size=224, num_classes=2, patch_size=16, latent_dim=opt.ermfc_input_dim, zero_head=True).cuda()
            encoder.load_from(np.load(opt.vit_path))
            regressor    = regressor_AL(opt.ermfc_input_dim).cuda()

        ermfc = load_ERM_FC(opt.ermfc_input_dim, opt.ermfc_output_dim).cuda()
        cha   = load_Cross_Attention(opt.ermfc_input_dim).cuda()

        e_opt = torch.optim.Adam(encoder.parameters(),     lr = opt.e_lr,      betas = (0.5, 0.99))
        m_opt = torch.optim.Adam(ermfc.parameters(),       lr = 0.2*opt.e_lr,  betas = (0.5, 0.99))
        r_opt = torch.optim.Adam(regressor.parameters(),   lr = opt.r_lr,      betas = (0.5, 0.99))
        c_opt = torch.optim.Adam(cha.parameters(),         lr = opt.e_lr,      betas = (0.5, 0.99))

        self.encoder   = encoder
        self.ermfc     = ermfc
        self.regressor = regressor
        self.cha       = cha

        self.e_opt = e_opt
        self.m_opt = m_opt
        self.r_opt = r_opt
        self.c_opt = c_opt

        self.e_lr_scheduler = lr_scheduler.MultiStepLR(self.e_opt, milestones=[5e2,25e2,50e2,75e2,100e2], gamma=0.8)
        self.m_lr_scheduler = lr_scheduler.MultiStepLR(self.m_opt, milestones=[5e2,25e2,50e2,75e2,100e2], gamma=0.8)
        self.r_lr_scheduler = lr_scheduler.MultiStepLR(self.r_opt, milestones=[5e2,25e2,50e2,75e2,100e2], gamma=0.8)
        self.c_lr_scheduler = lr_scheduler.MultiStepLR(self.c_opt, milestones=[5e2,25e2,50e2,75e2,100e2], gamma=0.8)

    def validate(self, current_info, n_iter):
        opt = self.opt
        MSE = nn.MSELoss()
        cnt = 0

        current_dir, current_time = current_info[0], current_info[1]
        self.encoder.eval()
        self.ermfc.eval()
        self.regressor.eval()

        rmse_v, rmse_a = 0., 0.
        prmse_v, prmse_a = 0., 0.
        inputs_list, all_z_list, scores_list, labels_list = [], [], [], []
        with torch.no_grad():
            for _, data_i in enumerate(self.val_loader):
    
                data, emotions = data_i['image'], data_i['va']
                valence = np.expand_dims(np.asarray(emotions[0]), axis=1)
                arousal = np.expand_dims(np.asarray(emotions[1]), axis=1)
                emo_np = np.concatenate([valence, arousal], axis=1)
                emotions = torch.from_numpy(emo_np).float()

                inputs, correct_labels = Variable(data.cuda()), Variable(emotions.cuda())

                if opt.model == 'mlpmixer':
                    _, all_z = self.encoder(inputs)
                    scores = self.ermfc(all_z)
                else:
                    all_z = self.regressor(self.encoder(inputs))
                    scores = self.ermfc(all_z)

                inputs_list.append(inputs.detach().cpu().numpy())
                all_z_list.append(all_z.detach().cpu().numpy())
                scores_list.append(scores.detach().cpu().numpy())
                labels_list.append(correct_labels.detach().cpu().numpy())
    
                RMSE_valence = MSE(scores[:,0], correct_labels[:,0])**0.5
                RMSE_arousal = MSE(scores[:,1], correct_labels[:,1])**0.5
    
                rmse_v += RMSE_valence.item(); rmse_a += RMSE_arousal.item()
                cnt = cnt + 1

        if n_iter % opt.print_check == 0:
            print(fg256("cyan", '\n[INFO] Images and features for EVAL save.'))
            all_z_th  = np.concatenate(all_z_list)
            scores_th = np.concatenate(scores_list)
            labels_th = np.concatenate(labels_list)

            np.save(opt.save_path+'eval_all_z_{}.npy'.format(n_iter), all_z_th)
            np.save(opt.save_path+'eval_scores_{}.npy'.format(n_iter), scores_th)
            np.save(opt.save_path+'eval_labels_{}.npy'.format(n_iter), labels_th)

            if n_iter == opt.print_check:
                inputs_th = np.concatenate(inputs_list)
                np.save(opt.save_path+'eval_inputs.npy', inputs_th)


        PCC_v, PCC_a, CCC_v, CCC_a, SAGR_v, SAGR_a, final_rmse_v, final_rmse_a = \
                metric_computation([rmse_v,rmse_a], scores_list, labels_list, cnt)
    
        # write results to log file
        if n_iter == opt.print_check:
            with open(current_dir+'/log/'+current_time+'.txt', 'a') as f:
                f.writelines(['{}\n\n'.format(opt)])

        with open(current_dir+'/log/'+current_time+'.txt', 'a') as f:
            f.writelines(['Itr:  \t {}, \
                    \nPCC:  \t{}|\t {}, \
                    \nCCC:  \t{}|\t {}, \
                    \nSAGR: \t{}|\t {}, \
                    \nRMSE: \t{}|\t {}\n\n'
                .format(
                    n_iter,
                    PCC_v[0,1],   PCC_a[0,1],
                    CCC_v[0,1],   CCC_a[0,1],
                    SAGR_v,       SAGR_a,
                    final_rmse_v, final_rmse_a,
            )])


    def train(self, current_info):
        opt = self.opt
        if opt.online_tracker:
            wandb.init(project=opt.project_title)

        MSE = nn.MSELoss()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        layer_norm = nn.LayerNorm(opt.ermfc_input_dim, elementwise_affine=False).cuda()

        n_iter = 0

        rng = default_rng()
        cnt, print_check, margin = 0, 50, 0.5

        self.encoder.train()
        self.regressor.train()
        self.ermfc.train()

        age_list = ['18', '30', '40', '50', '60', '85']  # you can arbitrary set

        for epoch in range(opt.num_epoch):
            print('epoch ' + str(epoch) + '/' + str(opt.num_epoch-1))

            epoch_iterator = tqdm(self.train_loader,
                                  desc="Training (X / X Steps) (loss=X.X)",
                                  bar_format="{l_bar}{r_bar}",
                                  dynamic_ncols=True)
            for _, data_i in enumerate(epoch_iterator):

                for enc_param_group in self.e_opt.param_groups:
                    aa = enc_param_group['lr']
                for reg_param_group in self.r_opt.param_groups:
                    bb = reg_param_group['lr']

                data, emotions, age = data_i['image'], data_i['va'], data_i['age']
                valence = np.expand_dims(np.asarray(emotions[0]), axis=1)
                arousal = np.expand_dims(np.asarray(emotions[1]), axis=1)
                emo_np = np.concatenate([valence, arousal], axis=1)
                emotions = torch.from_numpy(emo_np).float()
                age = np.asarray(age)

                ll_dictionary = dict()
                for k in range(len(age_list)):  # sort by each age group
                    ll_dictionary.update([[ age_list[k], [i for i,j in enumerate(age) if int(age_list[k-1]<=j and j<int(age_list[k])] ]])
        
                inputs, correct_labels = Variable(data.cuda()), Variable(emotions.cuda())


                # ---------------
                # Train regressor
                # ---------------
                if opt.model == 'mlpmixer':
                    _, latent_feats = self.encoder(inputs)
                    scores = self.ermfc(latent_feats)
                else:
                    latent_feats = self.regressor(self.encoder(inputs))
                    scores = self.ermfc(latent_feats)

                MSE_v = MSE(scores[:,0], correct_labels[:,0])
                MSE_a = MSE(scores[:,1], correct_labels[:,1])
    
                self.e_opt.zero_grad()
                self.r_opt.zero_grad()
                self.m_opt.zero_grad()

                pcc_loss, ccc_loss, ccc_v, ccc_a = pcc_ccc_loss(correct_labels, scores)

                loss = (MSE_v + MSE_a) + 0.5 * (pcc_loss + ccc_loss)
                loss.backward(retain_graph=True)
    
                self.e_opt.step()
                self.r_opt.step()
                self.m_opt.step()


                # ----------------------------------------
                # Domain quantification and loss functions
                # ----------------------------------------
                # warm-up
                if n_iter < opt.warmup_coef1:
                    temperature = 0.
                elif n_iter >= opt.warmup_coef1 and n_iter < opt.warmup_coef2:
                    temperature = n_iter / opt.warmup_coef2
                else:
                    temperature = 1.

                if temperature:
                    domain_label = np.arange(len(ll_sorted))
    
                    if opt.model == 'mlpmixer':
                        _, latent_variable = self.encoder(inputs)
                    else:
                        latent_variable = self.regressor(self.encoder(inputs))
    
                    try:
                        rnd_sample = rng.choice(domain_label, size=opt.no_domain, replace=False)
                    except ValueError:
                        rnd_sample = rng.choice(domain_label, size=2, repalce=False)
    
                    vector_dict = dict(); label_dict = dict()
                    labmn_dict  = dict()
                    domain_id_list = []
                    for abc in range(len(rnd_sample)):
                        
                        domain_id_list.append(abc)
    
                        mu = latent_variable[ ll_dictionary[ll_sorted[rnd_sample[abc]]] ]
                        label_gt = correct_labels[ ll_dictionary[ll_sorted[rnd_sample[abc]]] ]
                        label_mn = torch.FloatTensor([label_gt[i].norm(p=2) for i in range(label_gt.size(0))])
                        label_mn = F.normalize(label_mn.unsqueeze(0), p=1).log()  # normalize for log prob.
                        gumbel_sm = gumbel_softmax_sample(label_mn, 0.5)  # differentiable Gumbel Top-k reparametrization
                        vector_dict.update([[abc, mu]]); label_dict.update([[abc, label_gt]])
                        labmn_dict.update([[abc, gumbel_sm]])
    
                    # validity check
                    if len(domain_id_list) == 0:
                        continue
    
    
                    # ---------------
                    # Shape alignment
                    # ---------------
                    if opt.domain_sampling == 'gumbel-softmax':
                        for _, i in enumerate(domain_id_list):
                            _, indices = torch.topk(labmn_dict[i], opt.topk)
                            vector_dict[i] = vector_dict[i][indices].squeeze(0)
                            label_dict[i]  = label_dict[i][indices].squeeze(0)
                    elif opt.domain_sampling == 'max-filling':
                        user_no_list = [len(vector_dict[i]) for _, i in enumerate(domain_id_list)]
                        max_user_no = np.max(user_no_list)
                        for nnn, i in enumerate(domain_id_list):
                            user_iter = max_user_no - user_no_list[nnn]
                            if user_iter == 0: continue
                            else:
                                vector_dict[i] = torch.cat([vector_dict[i], vector_dict[i][-1].repeat(user_iter,1)], dim=0)
                                label_dict[i]  = torch.cat([label_dict[i], label_dict[i][-1].repeat(user_iter,1)], dim=0)
                    elif opt.domain_sampling == 'none':
                        pass
    
                    # quantifying ID shift
                    vectors = [vector_dict[i] for i in sorted(vector_dict)]
                    labels  = [label_dict[i] for i in sorted(label_dict)]
                    if opt.matching_method == 'Sinkhorn':
                        domain_shift = Sinkhorn_Knopp(opt, vectors, cos, sinkhorn=opt.sinkhorn)
                    elif opt.matching_method == 'Cross-attention':
                        domain_shift = Cross_Attention(opt, CHA, vectors)
    
    
                    # --------------------
                    # Domain-wise ERM loss
                    # --------------------
                    total_erm_loss, total_reg_loss = 0., 0.
                    erm_loss, reg_loss = 0., 0.
                    for nnn, usr_idx in enumerate(domain_id_list):
                        aaa = LENGTH_CHECK(vector_dict[usr_idx])
                        if usr_idx == domain_id_list[0]:
                            erm_loss = 0.
                            outputs = self.ermfc(aaa).float()
                            erm_loss = MSE(outputs, label_dict[usr_idx])
                        else:
                            upper = aaa - domain_shift[nnn-1][0]
                            lower = upper.pow(2).sum(dim=1) / opt.ermfc_input_dim
                            outputs = self.ermfc( upper / (torch.sqrt(lower.unsqueeze(1))+opt.epsilon) ).float()
                            erm_loss = MSE(outputs, label_dict[usr_idx])
                        total_erm_loss += erm_loss
    
                    total_loss = 0.5 * total_erm_loss
                    total_loss = temperature * total_loss
                    self.e_opt.zero_grad()
                    self.r_opt.zero_grad()
                    self.m_opt.zero_grad()
                    if opt.matching_method == 'Cross-attention':
                        self.c_opt.zero_grad()
    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)  # gradient clipping
    
                    self.e_opt.step()
                    self.r_opt.step()
                    self.m_opt.step()
                    if opt.matching_method == 'Cross-attention':
                        self.c_opt.step()
    
                    if opt.online_tracker:
                        wandb.log({
                            "loss": loss.item(),
                            "ERM loss": total_erm_loss.item(),
                            "Enc_lr": aa, "Reg_lr": bb,
                            "epoch": epoch, "ccc_v": ccc_v.item(), "ccc_a": ccc_a.item(),
                            "RMSE (v)": MSE_v.item(), "RMSE (a)": MSE_a.item(),
                        })
   
                if n_iter % opt.print_check == 0 and n_iter > 0:
                    torch.save(self.encoder.state_dict(),   opt.save_path+'encoder_{}_{}.t7'.format(n_iter, epoch))
                    torch.save(self.regressor.state_dict(), opt.save_path+'regressor_{}_{}.t7'.format(n_iter, epoch))
                    torch.save(self.ermfc.state_dict(), opt.save_path+'ermfc_{}_{}.t7'.format(n_iter, epoch))
                    self.validate(current_info, n_iter)

                n_iter = n_iter + 1

                self.e_lr_scheduler.step()
                self.r_lr_scheduler.step()
                self.m_lr_scheduler.step()
                self.c_lr_scheduler.step()
