import numpy as np

def metric_computation(rmse, scores, labels, cnt):

    rmse_v = rmse[0]
    rmse_a = rmse[1]

    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)

    std_l_v = np.std(labels[:,0]); std_p_v = np.std(scores[:,0])
    std_l_a = np.std(labels[:,1]); std_p_a = np.std(scores[:,1])

    mean_l_v = np.mean(labels[:,0]); mean_p_v = np.mean(scores[:,0])
    mean_l_a = np.mean(labels[:,1]); mean_p_a = np.mean(scores[:,1])

    PCC_v = np.cov(labels[:,0], np.transpose(scores[:,0])) / (std_l_v * std_p_v)
    PCC_a = np.cov(labels[:,1], np.transpose(scores[:,1])) / (std_l_a * std_p_a)
    CCC_v = 2.0 * np.cov(labels[:,0], np.transpose(scores[:,0])) / ( np.power(std_l_v,2) + np.power(std_p_v,2) + (mean_l_v - mean_p_v)**2 )
    CCC_a = 2.0 * np.cov(labels[:,1], np.transpose(scores[:,1])) / ( np.power(std_l_a,2) + np.power(std_p_a,2) + (mean_l_a - mean_p_a)**2 )

    sagr_v_cnt = 0
    for i in range(len(labels)):
        if np.sign(labels[i,0]) == np.sign(scores[i,0]) and labels[i,0] != 0:
            sagr_v_cnt += 1
    SAGR_v = sagr_v_cnt / len(labels)

    sagr_a_cnt = 0
    for i in range(len(labels)):
        if np.sign(labels[i,1]) == np.sign(scores[i,1]) and labels[i,1] != 0:
            sagr_a_cnt += 1
    SAGR_a = sagr_a_cnt / len(labels)

    final_rmse_v = rmse_v/cnt
    final_rmse_a = rmse_a/cnt
    return PCC_v, PCC_a, CCC_v, CCC_a, SAGR_v, SAGR_a, final_rmse_v, final_rmse_a
