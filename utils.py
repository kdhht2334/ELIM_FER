import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function, Variable

from fabulous.color import fg256


def LENGTH_CHECK(vec):
    if len(vec) == 1:
        return torch.cat([vec, vec], dim=0)
    else:
        return vec


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def Cross_Attention(args, model, vectors):

    CHA = model
    shift_list = []
    no_domain = len(vectors)
    anchor_vec = vectors[0]

    for user in range(no_domain-1):  # one-to-N pair-based cross attention
        weights = CHA(anchor_vec.unsqueeze(0), vectors[user+1].unsqueeze(0))
        shift_list.append(weights)

    return shift_list


def Sinkhorn_Knopp(args, vectors, metric, sinkhorn=True):

    cos = metric
    no_domain = len(vectors)
    mean = [torch.mean(vectors[i], dim=0, keepdim=True) for i in range(no_domain)]

    # weights of OT
    sim_user_anchor = torch.zeros(size=(no_domain-1, vectors[0].size(0),1))
    sim_user_list = []
    for user in range(no_domain-1):
        sim_user_list.append( torch.zeros(size=(1,vectors[user+1].size(0),1)) )
    anchor_vec = vectors[0]
    anchor_mean = mean[0]

    if args.relevance_weighting == 0:
        for user in range(no_domain-1):
            gen = (vector.norm(p=2) for vector in anchor_vec)
            for idx, sim in enumerate(gen):
                sim_user_anchor[user,idx,0] = sim
            gen = (vector.norm(p=2) for vector in vectors[user+1])
            for idx, sim in enumerate(gen):
                sim_user_list[user][0,idx,0] = sim
    elif args.relevance_weighting == 1:
        for user in range(no_domain-1):
            gen = (vector @ mean[user+1].t() for vector in anchor_vec)
            for idx, sim in enumerate(gen):
                sim_user_anchor[user,idx,0] = F.relu(sim)
            gen = (vector @ anchor_mean.t() for vector in vectors[user+1])
            for idx, sim in enumerate(gen):
                sim_user_list[user][0,idx,0] = F.relu(sim)

    sim_user_anchor = (sim_user_anchor.size(1)*sim_user_anchor) / (torch.sum(sim_user_anchor,1).unsqueeze(1)+args.epsilon)
    for user in range(no_domain-1):
        sim_user_list[user] = (sim_user_list[user].size(1)*sim_user_list[user]) / (torch.sum(sim_user_list[user],1).unsqueeze(1))

    # cost of OT
    cos_mat_list = []
    for user in range(no_domain-1):
        cos_mat_list.append(torch.zeros(size=(1,vectors[0].size(0),vectors[user+1].size(0))))
    for user in range(no_domain-1):
        for left in range(cos_mat_list[user].size(1)):
            for right in range(cos_mat_list[user].size(2)):
                cos_mat_list[user][0,left,right] = 1. - cos(vectors[0][left].unsqueeze_(0), vectors[user+1][right].unsqueeze_(0))

    if sinkhorn:
        _lambda, _scale_factor, _no_iter = 5., 0.1, 5
        scale_list, shift_list = [], []
        for user in range(no_domain-1):  # repeat for each identity
            r = sim_user_anchor[user]
            c = sim_user_list[user].squeeze(0)
            M = cos_mat_list[user].squeeze(0)

            u = torch.ones(size=(r.size(0),1))/r.size(0)
            K = torch.exp(-_lambda*M)
            K_tilde = torch.diag(1/(r[:,0]+args.epsilon)) @ K

            # update u,v
            for itrs in range(_no_iter):
                u_new = 1 / ( K_tilde @ (c/(K.t()@u+args.epsilon))+args.epsilon )
                u = u_new
            v = c / (K.t()@u+args.epsilon)

            apprx_flow = torch.diag(u.squeeze(1)) @ K @ torch.diag(v.squeeze(1))
            MMM = apprx_flow * M
            mu_e = torch.sum(MMM, dim=0, keepdim=False).unsqueeze(1)
            shift_list.append(mu_e.detach().cuda())
        return shift_list


def pcc_ccc_loss(labels_th, scores_th):
    std_l_v = torch.std(labels_th[:,0]); std_p_v = torch.std(scores_th[:,0])
    std_l_a = torch.std(labels_th[:,1]); std_p_a = torch.std(scores_th[:,1])
    mean_l_v = torch.mean(labels_th[:,0]); mean_p_v = torch.mean(scores_th[:,0])
    mean_l_a = torch.mean(labels_th[:,1]); mean_p_a = torch.mean(scores_th[:,1])
    
    PCC_v = torch.mean( (labels_th[:,0] - mean_l_v) * (scores_th[:,0] - mean_p_v) ) / (std_l_v * std_p_v)
    PCC_a = torch.mean( (labels_th[:,1] - mean_l_a) * (scores_th[:,1] - mean_p_a) ) / (std_l_a * std_p_a)
    CCC_v = (2.0 * std_l_v * std_p_v * PCC_v) / ( std_l_v.pow(2) + std_p_v.pow(2) + (mean_l_v-mean_p_v).pow(2) )
    CCC_a = (2.0 * std_l_a * std_p_a * PCC_a) / ( std_l_a.pow(2) + std_p_a.pow(2) + (mean_l_a-mean_p_a).pow(2) )
    
    PCC_loss = 1.0 - (PCC_v + PCC_a)/2
    CCC_loss = 1.0 - (CCC_v + CCC_a)/2
    return PCC_loss, CCC_loss, CCC_v, CCC_a

