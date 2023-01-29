import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import Counter
from copy import deepcopy
import argparse
import time
from argparse import ArgumentParser
# from . import read_data
from utils.read_data import preprocessing, obtain_group_index, obtain_group_index_tl
from scipy.special import softmax
from tqdm import tqdm, trange
from utils.Disparity_Metrics import *
import json


def parser_args():
    parser = ArgumentParser(description="JMEF")
    parser.add_argument('--data', type=str, default='ml-1m', choices=['ml-1m', 'ml-100k', 'lt'],
                        help="File path for data")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0, help="Seed (For reproducability)")
    parser.add_argument('--model', type=str, default='Pop')
    parser.add_argument('--gamma', type=float, default=0.8, help="patience factor")
    parser.add_argument('--temp', type=float, default=0.1, help="temperature. how soft the ranks to be")
    parser.add_argument('--s_ep', type=int, default=100)
    parser.add_argument('--r_ep', type=int, default=1)
    parser.add_argument('--norm', type=str, default='N')
    parser.add_argument('--coll', type=str, default='Y')
    parser.add_argument('--age', type=str, default='N')
    parser.add_argument('--ndatapoints', type=int, default= 5000)
    parser.add_argument('--conduct', type=str, default='sh')
    return parser.parse_args()


def normalize_matrix_by_row(matrix):
    """ Normalize matrix per row
    Input: 
        matrix: matrix to be normalized
    Output:
        normalized_matrix: input matrix normalized by row
    """
    sum_of_rows = matrix.sum(axis=1)
    normalized_matrix = matrix / sum_of_rows[:, np.newaxis]
    return normalized_matrix

def calc_num_rel(matrix_label):
    """ Calculate the 
    Input:
        matrix_label: matrix of labels
    Output:
        num_rel: 
    """
    num_rel = matrix_label.sum(1, keepdims=True).reshape(-1, 1).astype("float")#Y sum in 1st dim of rating matrix 6040x1
    num_rel[num_rel == 0.0] = 1.0
    
    return num_rel


def calc_E_target(args, matrix_label, num_rel):
    """ Calculate E_target with user browsing model (USM)
    """
    usm_exposure = (args.gamma / (1.0 - args.gamma)) * (1.0 - np.power(args.gamma, num_rel).astype("float")) #6040x1
    E_target = usm_exposure / num_rel * matrix_label #[6040,3706]
    return E_target

def build_E_collect(E_target):
    """ Calculate E_collect
    """
    if args.coll == 'Y':
        E_collect = np.ones((E_target.shape[0], E_target.shape[1])) * E_target.mean() #[6040,3706]
    else:
        E_collect = np.zeros((E_target.shape[0], E_target.shape[1]))
    return E_collect

def calc_E_system(args, E_target, top_item_id, weight = np.nan):
    """ Calculate E_system
    """

    E_system = np.zeros((E_target.shape[0], E_target.shape[1]))
    
    if args.conduct == 'st':
        exp_vector = np.power(args.gamma, np.arange(100) + 1).astype("float")
        for i in range(len(top_item_id)):
            top_item_id = [list(map(int, i)) for i in top_item_id]
            E_system[i][top_item_id[i]] = exp_vector
        
        return  torch.from_numpy(E_system)
    
    if args.conduct == 'sh':
        sample_times = args.s_ep
        for sample_epoch in trange(sample_times, ascii=False): # sample 100 rankings for each user 
            E_system_tmp = np.zeros((E_target.shape[0], E_target.shape[1]))
            exp_vector = np.power(args.gamma, np.arange(100) + 1).astype("float")  # pre-compute the exposure_vector (100x1)
            for i in range(len(top_item_id)):
                tmp_selected = np.random.choice(top_item_id[i], 100, replace=False, p=weight[i]) #selects one permutation of 100 movies from /
                #top 100 movies from a user's rank with probability weights[user] (100x1)
                
                tmp_selected = np.array([int(j) for j in tmp_selected])
                E_system_tmp[i][tmp_selected] = exp_vector
            E_system += E_system_tmp
        E_system /= sample_times

        return torch.from_numpy(E_system)

def eval_function_stochas(save_df, user_label, item_label, matrix_label, args, rand_tau=1):
    # construct E_target
    num_rel = calc_num_rel(matrix_label)
    
    # user browsing model  
    E_target = calc_E_target(args, matrix_label, num_rel)

    # construct E_collect
    E_collect = build_E_collect(E_target)

    # To pytorch tensors 
    E_target = torch.from_numpy(E_target)
    E_collect = torch.from_numpy(E_collect)

    print(len(save_df['item']))
    top_item_id = np.array(list(save_df["item"])).reshape(-1, 100) #[6040, 100]
    top_score = np.array(list(save_df["score"])).reshape(-1, 100)
    print('top_item_id', top_item_id.shape)
    print('top score ', top_score.shape)
    if args.norm == 'Y':
        top_score = normalize_matrix_by_row(top_score)
    weight = softmax(top_score / rand_tau, axis=1) #Y/b in quation of p(d|u)
    
    indicator = torch.ones((E_target.shape[0], E_target.shape[1]))

    E_system = calc_E_system(args, E_target, top_item_id, weight=weight)

    IIF_all = II_F(E_system, E_target, E_collect, indicator)
    GIF_all = GI_F(E_system, E_target, E_collect, user_label, indicator)
    AIF_all = AI_F(E_system, E_target, E_collect, indicator)
    IGF_all = IG_F(E_system, E_target, E_collect, item_label, indicator)
    GGF_all = GG_F(E_system, E_target, E_collect, user_label, item_label, indicator)[:3]
    AGF_all = AG_F(E_system, E_target, E_collect, item_label, indicator)
    print('Metric evaluation complete')
    return IIF_all, GIF_all, IGF_all, GGF_all, AIF_all, AGF_all  # , IIF_sp, IGF_sp, GIF_sp, GGF_sp, AIF_sp, AGF_sp
    

def eval_function_static(save_df, user_label, item_label, matrix_label, args):
    # construct E_target
    num_rel = calc_num_rel(matrix_label)

    # Calculate E_target with user browsing model
    E_target = calc_E_target(args, matrix_label, num_rel)

    # construct E_collect (collectiion of exposures?), E_collect = random exposure
    E_collect = build_E_collect(E_target)

    top_item_id = np.array(list(save_df["item"])).reshape(-1, 100)
    
    # put the exposure value into the selected positions
    E_system = calc_E_system(args, E_target, top_item_id)

    E_target = torch.from_numpy(E_target)
    E_collect = torch.from_numpy(E_collect)
    indicator = torch.ones((E_target.shape[0], E_target.shape[1]))
    IIF = II_F(E_system, E_target, E_collect, indicator)
    GIF = GI_F(E_system, E_target, E_collect, user_label, indicator)
    AIF = AI_F(E_system, E_target, E_collect, indicator)
    IGF = IG_F(E_system, E_target, E_collect, item_label, indicator)
    GGF = GG_F(E_system, E_target, E_collect, user_label, item_label, indicator)[:3]
    AGF = AG_F(E_system, E_target, E_collect, item_label, indicator)

    return IIF, GIF, IGF, GGF, AIF, AGF


def compute_stochas(args):
    save_df = load_deterministic_ranker(args)
    if args.model == 'LDA':
        save_df["score"] = save_df["score"] * 1000
    elif args.model == 'Pop':
        save_df["score"] = save_df["score"] * 10
    elif args.model in ["PLSA", "RM1", "RSV", "CHI2", "HT", "KLD", "SVD", "UIR", "RM2", "LMWU", "LMWI", "NNU", "NNI"]:
        args.norm = 'Y'
    
    # List of beta_values
    rand_tau_list = [8, 4, 2, 1, 0.5, 0.25, 0.125] # different values for beta
    
    save_IIF, save_IGF, save_GIF, save_GGF, save_AIF, save_AGF = [], [], [], [], [], []
    save_IID, save_IGD, save_GID, save_GGD, save_AID, save_AGD = [], [], [], [], [], []
    save_IIR, save_IGR, save_GIR, save_GGR, save_AIR, save_AGR = [], [], [], [], [], []

    len_tau = len(rand_tau_list)

    """evaluate on whole"""
    for epoch in range(args.r_ep):
        print("epoch:", epoch)
        for i in range(len_tau):
            print("tau={}".format(rand_tau_list[i]))

            # IIF_all, GIF_all, IGF_all, GGF_all, AIF_all, AGF_all, IIF_sp, IGF_sp, GIF_sp, GGF_sp, AIF_sp, AGF_sp \
            IIF_all, GIF_all, IGF_all, GGF_all, AIF_all, AGF_all \
                = eval_function_stochas(save_df, user_label, item_label, matrix_label, args, rand_tau=rand_tau_list[i])

            save_IIF.append(IIF_all[0].item())
            save_GIF.append(GIF_all[0].item())
            save_IGF.append(IGF_all[0].item())
            save_GGF.append(GGF_all[0].item())
            save_AIF.append(AIF_all[0].item())
            save_AGF.append(AGF_all[0].item())
 
            save_IID.append(IIF_all[1].item())
            save_GID.append(GIF_all[1].item())
            save_IGD.append(IGF_all[1].item())
            save_GGD.append(GGF_all[1].item())
            save_AID.append(AIF_all[1].item())
            save_AGD.append(AGF_all[1].item())

            save_IIR.append(IIF_all[2].item())
            save_GIR.append(GIF_all[2].item())
            save_IGR.append(IGF_all[2].item())
            save_GGR.append(GGF_all[2].item())
            save_AIR.append(AIF_all[2].item())
            save_AGR.append(AGF_all[2].item())

    dict_all = {"IIF": save_IIF, "IGF": save_IGF, "GIF": save_GIF, "GGF": save_GGF, "AIF": save_AIF, "AGF": save_AGF,
                "IID": save_IID, "IGD": save_IGD, "GID": save_GID, "GGD": save_GGD, "AID": save_AID, "AGD": save_AGD,
                "IIR": save_IIR, "IGR": save_IGR, "GIR": save_GIR, "GGR": save_GGR, "AIR": save_AIR, "AGR": save_AGR}

    # Save files in json format
    for key in dict_all:
        if args.age == 'Y':
            with open("src/outputs/{}/{}_all_{}_Y.json".format(args.data, key, args.model), "w") as fp:
                json.dump(dict_all[key], fp)
        else:
            with open(
                    "src/outputs/{}/{}_all_{}.json".format(args.data, key, args.model), "w") as fp:
                json.dump(dict_all[key], fp)
    

    return dict_all


def compute_static(args):
    # Load deterministic ranker
    save_df = load_deterministic_ranker(args)

    save_IIF, save_IGF, save_GIF, save_GGF, save_AIF, save_AGF = [], [], [], [], [], []
    save_IID, save_IGD, save_GID, save_GGD, save_AID, save_AGD = [], [], [], [], [], []
    save_IIR, save_IGR, save_GIR, save_GGR, save_AIR, save_AGR = [], [], [], [], [], []

    IIF_all, GIF_all, IGF_all, GGF_all, AIF_all, AGF_all \
        = eval_function_static(save_df, user_label, item_label, matrix_label, args)

    save_IIF.append(IIF_all[0].item())
    save_GIF.append(GIF_all[0].item())
    save_IGF.append(IGF_all[0].item())
    save_GGF.append(GGF_all[0].item())
    save_AIF.append(AIF_all[0].item())
    save_AGF.append(AGF_all[0].item())

    save_IID.append(IIF_all[1].item())
    save_GID.append(GIF_all[1].item())
    save_IGD.append(IGF_all[1].item())
    save_GGD.append(GGF_all[1].item())
    save_AID.append(AIF_all[1].item())
    save_AGD.append(AGF_all[1].item())

    save_IIR.append(IIF_all[2].item())
    save_GIR.append(GIF_all[2].item())
    save_IGR.append(IGF_all[2].item())
    save_GGR.append(GGF_all[2].item())
    save_AIR.append(AIF_all[2].item())
    save_AGR.append(AGF_all[2].item())

    print("save_IIF:", save_IIF)
    print("save_IID:", save_IID)
    print("save_IIR:", save_IIR)

    dict = {"IIF": save_IIF, "IGF": save_IGF, "GIF": save_GIF, "GGF": save_GGF, "AIF": save_AIF, "AGF": save_AGF,
            "IID": save_IID, "IGD": save_IGD, "GID": save_GID, "GGD": save_GGD, "AID": save_AID, "AGD": save_AGD,
            "IIR": save_IIR, "IGR": save_IGR, "GIR": save_GIR, "GGR": save_GGR, "AIR": save_AIR, "AGR": save_AGR}

    for key in dict:
        if args.age == 'Y':
            with open("src/outputs/{}/{}_all_{}_static_Y.json".format(args.data, key, args.model), "w") as fp:
                json.dump(dict[key], fp)
        else:
            with open("src/outputs/{}/{}_all_{}_static.json".format(args.data, key, args.model), "w") as fp:
                json.dump(dict[key], fp)

def load_deterministic_ranker(args):
    """ Load pretrained deterministic ranker 
    """
    
    if (args.data == 'ml-1m') or (args.data == 'ml-100k'):
        save_df = pd.read_csv('src/models/ml/run-{}-ml-1M-fold1.txt.gz'.format(args.model),
                          compression='gzip', header=None, sep='\t', quotechar='"', usecols=[0, 2, 4])
    elif args.data == 'lt':
        save_df = pd.read_csv('src/models/runs-libraryThing/run-{}-libraryThing-fold1.txt.gz'.format(args.model),
                          compression='gzip', header=None, sep='\t', quotechar='"', usecols=[0, 2, 4])
    

    save_df = save_df.rename(columns={0: "user", 2: "item", 4: "score"})
    if (args.data == 'ml-1m') or (args.data == 'ml-100k'):
        save_df.user = save_df.user - 1

    save_df['item'] = save_df['item'].map(item_mapping)
    save_df = save_df.dropna().reset_index(drop = True)
    save_df.drop(save_df.tail(len(save_df)%100).index, inplace = True)
    
    
    save_df = save_df.sort_values(["user", "score"], ascending=[True, False])
    save_df = save_df.reset_index().drop(['index'], axis=1)
    return save_df



def compute_exp_matrix(args):
    save_df = load_deterministic_ranker(args)

    if args.model == 'LDA':
        save_df["score"] = save_df["score"] * 1000
    save_IIF, save_IGF, save_GIF, save_GGF, save_AIF, save_AGF = [], [], [], [], [], []
    save_IID, save_IGD, save_GID, save_GGD, save_AID, save_AGD = [], [], [], [], [], []
    save_IIR, save_IGR, save_GIR, save_GGR, save_AIR, save_AGR = [], [], [], [], [], []

    # rand_tau_list = [2, 4, 8, 16]
    rand_tau_list = [0.125, 8]
    len_tau = len(rand_tau_list)

    """evaluate on whole"""
    for i in range(len_tau):
        rand_tau = rand_tau_list[i]
        print("tau={}".format(rand_tau))
        # construct E_target
        num_rel = calc_num_rel(matrix_label)

        # Calculate E_target with user browsing model
        E_target = calc_E_target(args, matrix_label, num_rel)

        # construct E_collect
        E_collect = np.ones((E_target.shape[0], E_target.shape[1])) * E_target.mean()


        top_item_id = np.array(list(save_df["item"])).reshape(-1, 100)
        top_score = np.array(list(save_df["score"])).reshape(-1, 100)
        # This was commented out at some point
        top_score = normalize_matrix_by_row(top_score)
        weight = softmax(top_score / rand_tau, axis=1)

        # put the exposure value into the selected positions
        sample_times = 100
        E_system = np.zeros((E_target.shape[0], E_target.shape[1]))
        for _ in trange(sample_times, ascii=False):
            E_system_tmp = np.zeros((E_target.shape[0], E_target.shape[1]))
            exp_vector = np.power(args.gamma, np.arange(100) + 1).astype("float")
            for i in range(len(top_item_id)):
                tmp_selected = np.random.choice(top_item_id[i], 100, replace=False, p=weight[i])
                tmp_selected = np.array([int(j) for j in tmp_selected])
                E_system_tmp[i][tmp_selected] = exp_vector
            E_system += E_system_tmp
        E_system /= sample_times

        E_system = torch.from_numpy(E_system)
        E_target = torch.from_numpy(E_target)
        E_collect = torch.from_numpy(E_collect)
        indicator = torch.ones((E_target.shape[0], E_target.shape[1]))
        GG_target_stochas = GG_F(E_system, E_target, E_collect, user_label, item_label, indicator)[3]
        GG_system_stochas = GG_F(E_system, E_target, E_collect, user_label, item_label, indicator)[4]

        with open("src/outputs/{}/GG_MT_{}_{}.json".format(args.data, rand_tau, args.model), "w") as fp:
            json.dump(np.array(GG_target_stochas).tolist(), fp)
        with open("src/outputs/{}/GG_MS_{}_{}.json".format(args.data, rand_tau, args.model), "w") as fp:
            json.dump(np.array(GG_system_stochas).tolist(), fp)

    # construct E_target
    num_rel = calc_num_rel(matrix_label)

    exposure_rel = (args.gamma / (1.0 - args.gamma)) * (1.0 - np.power(args.gamma, num_rel).astype("float"))
    E_target = exposure_rel / num_rel * matrix_label

    # construct E_collect
    E_collect = np.ones((E_target.shape[0], E_target.shape[1])) * E_target.mean()

    # construct E_system
    user_size = E_target.shape[0]

    top_item_id = np.array(list(save_df["item"])).reshape(-1, 100)
    top_score = np.array(list(save_df["score"])).reshape(-1, 100)

    # put the exposure value into the selected positions
    E_system = np.zeros((E_target.shape[0], E_target.shape[1]))
    exp_vector = np.power(args.gamma, np.arange(100) + 1).astype("float")
    for i in range(len(top_item_id)):
        top_item_id = [list(map(int, i)) for i in top_item_id]
        E_system[i][top_item_id[i]] = exp_vector

    E_system = torch.from_numpy(E_system)
    E_target = torch.from_numpy(E_target)
    E_collect = torch.from_numpy(E_collect)
    indicator = torch.ones((E_target.shape[0], E_target.shape[1]))
    GG_target_static = GG_F(E_system, E_target, E_collect, user_label, item_label, indicator)[3]
    GG_system_static = GG_F(E_system, E_target, E_collect, user_label, item_label, indicator)[4]
    GG_collect = GG_F(E_system, E_target, E_collect, user_label, item_label, indicator)[5]

    print("GG_target_static:", GG_target_static)
    print("GG_system_static:", GG_system_static)
    print("GG_collect:", GG_collect)

    with open("src/outputs/{}/GG_MT_{}_static.json".format(args.data, args.model), "w") as fp:
        json.dump(np.array(GG_target_static).tolist(), fp)
    with open("src/outputs/{}/GG_MS_{}_static.json".format(args.data, args.model), "w") as fp:
        json.dump(np.array(GG_system_static).tolist(), fp)
    with open("src/outputs/{}/GG_collect_{}_static.json".format(args.data, args.model), "w") as fp:
        json.dump(np.array(GG_collect).tolist(), fp)


if __name__ == '__main__':
    print('start timer')
    start = time.time()

    args = parser_args()
    args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("device:", args.device)

    """read data and assign attribute index"""
    df, item_mapping, matrix_label, user_size, item_size = preprocessing(args)
    
    # Obtain group indices for Movielens dataset
    if (args.data == 'ml-1m') or (args.data == 'ml-100k'):
        index_F, index_M, index_gender, index_age, index_genre, index_pop, index_occup, age_matrix, pop_mask, occup_matrix, genre_matrix \
            = obtain_group_index(df, args)
        
        # Set user group lable:
        if args.age == 'Y':
            user_label = age_matrix #[7,6040]
        elif args.age == 'Occup':
            user_label = occup_matrix
        else:
            # Build matrix with gender information
            gender_matrix = torch.zeros(2, len(index_F) + len(index_M)) #[2, #females + #males] , 1st row for F 2nd for M
            for ind in index_F:
                gender_matrix[0][ind] = 1
            for ind in index_M:
                gender_matrix[1][ind] = 1
    
            user_label = gender_matrix  # .to(args.device) [2,6040]

        # Set item group lable
        item_label = genre_matrix  # .to(args.device) [18, 3706]

    # Obtain group indices for LibraryThing dataset
    elif (args.data == 'lt'):
        index_engagement, index_helpful, engagement_matrix, helpful_matrix = obtain_group_index_tl(df, args) 
        user_label = helpful_matrix 
        item_label = engagement_matrix


    matrix_label = np.array(matrix_label.todense()) #rating matrix for matrix factorization, user-item relevance matrix Y [6040, 3706]
    print('mat lab', matrix_label.shape)
    # Print set-up statics
    print('-------- Configuration of Experiment --------')
    print("norm:", args.norm)
    print("coll:", args.coll)
    print("model:", args.model)
    print('conduct:', args.conduct)
    print('----------------------------------------------')

    # Run computation according to conduct nide
    if args.conduct == 'sh':
        compute_stochas(args)
    elif args.conduct == 'st':
        compute_static(args)

    # Compute Expectation matrix
    #compute_exp_matrix(args)

    stop = time.time()
    print('Time elapsed: ', np.round(stop-start, 4), ' s')
    
    
    # Write experiment GPU time to file
    with open("Outputs/Experiment_times.txt", "a") as f:
        f.write(args.model + " " + args.age + " " + args.conduct + " " + str(np.round(stop-start, 4)) + " s  \n")
    
    print('---------------------------------------------')
    print("Experiment completed successfully")
    print('---------------------------------------------')