import pandas as pd
import numpy  as np
import random
import time
import collections
from pandas import Series, DataFrame
from tqdm import tqdm, trange
import operator
from functools import reduce
import copy
from itertools import chain
from matplotlib import pyplot
import matplotlib.pyplot as plt
from tkinter import _flatten
from IPython.core.pylabtools import figsize
import scipy.stats as stats
from multiprocessing import Pool
import IjcaiCalReward
import calArmFeatures
import structUser
import sendMail
import KMaxValue


def updateUserParams(u, x, reward):
    user_index = allUsers.user_parameters[allUsers.user_parameters['user_id'] == u].index[0]
    Mu = allUsers.user_parameters.at[user_index, 'M']
    bu = allUsers.user_parameters.at[user_index, 'b']
    Mu = Mu + x.dot(x.T)
    bu = bu + reward * x
    allUsers.user_parameters.at[user_index, 'M'] = Mu
    allUsers.user_parameters.at[user_index, 'b'] = bu


def update_user_bandit(u, similarUsers, users, reward):
    uIndex = users.index(u)
    for i in similarUsers:
        similarUsersIndex = users.index(i)
        if reward:
            BetaA[uIndex][similarUsersIndex] += 1
            BetaA[similarUsersIndex][uIndex] += 1
        else:
            BetaB[uIndex][similarUsersIndex] += 1
            BetaB[similarUsersIndex][uIndex] += 1


def findNeighborhood(users, u, gap):
    # index of user u
    uIndex = users.index(u)
    # caculate the beta for each user
    resBeta = [0] * len(users)
    kIndex = []
    for i in range(len(users)):
        a = BetaA[uIndex][i]
        b = BetaB[uIndex][i]
        resBeta[i] = np.random.beta(a, b)
        if (resBeta[i] >= gap):
            kIndex.append(i)
    # find the k max values of resBeta and the corresponding users
    res = np.array(users)
    similiarUser = res[kIndex].tolist()
    return similiarUser


def Neighbor(users, u, round, gap):
    size = len(data[data['user_id'] == u])  
    u_sequence = data[data['user_id'] == u].reset_index(drop=True)
    u_interactive_sequence = u_sequence.iloc[round, :]
    Hit = [0]  # hit items at i rounds for u
    res = [0] * arm_num  
    recom_len = list(set(list(data['item_id'][data['user_id'] == u].values)))
    arms = recom_len
    F1 = 0
    recall = 0
    precision = 0
    a = 0.1
    user_index = allUsers.user_parameters[allUsers.user_parameters['user_id'] == u].index[0]
    # find the kneighborhoods of the target user
    similarUsers = findNeighborhood(users, u, gap)
    # print("相似用户为", similarUsers)
    # caculate the parameters of neighborhoods
    cluster_M, cluster_b, cluster_w = allUsers.caculate_cluster_parameters(similarUsers)
    # item partitioning
    remain_len = arm_num - len(recom_len)
    remains = random.sample(total_item, remain_len)
    arms.extend(remains)
    random.shuffle(arms)
    # calculate context features of arms
    for i in range(arm_num):
        single_item = arms[i]
        arm_feature = arm_context.find_item_feature(single_item, item_features, d)
        temp1 = (arm_feature.T).dot(np.linalg.inv(cluster_M)).dot(arm_feature)
        temp = a * np.sqrt(temp1[0][0] * np.log(round + 1))
        res[i] = (cluster_w.T).dot(arm_feature) + temp
    # find the top k of res
    kMax = KMaxValue.kMaxValues()
    action_index = kMax.kMaxValues(res, N)
    arms1 = np.array(arms)
    recommend_list_N = arms1[action_index]
    total_R = 0
    if len(recommend_list_N):
        Hit = []
        for item in recommend_list_N:
            s, flag = payoff.get_reward(u_interactive_sequence, item, data)
            total_R += s
            if flag:
                Hit.append(item)
    contxtFeatures = arm_context.cal_arm_features(recommend_list_N, item_features, d)
    updateUserParams(u, contxtFeatures, total_R / N)
    # update the user bandit
    update_user_bandit(u, similarUsers, users, total_R)
    # sumarize the metrics
    recall = len(set(Hit)) / len(set(recom_len))
    precision = len(set(Hit)) / len(set(recommend_list_N))
    if (recall or precision):
        F1 = (2 * recall * precision) / (recall + precision)
    return precision, recall, F1, total_R


def main(testUsersNum, rounds, recomLength, dimension, armNum, gap, alpha):
    global data
    global item_features
    global total_item
    global allUsers
    global d
    global N
    global arm_num
    global arm_context
    global payoff
    global BetaA
    global BetaB
    N = recomLength
    d = dimension
    arm_num = armNum
    data = pd.read_csv('Data/IjcaiData.csv')
    item_features = pd.read_csv('Data/IjcaiArmFeatures' + str(d) + '.csv')
    total_item = item_features['item_id'].values.tolist()
    test_users = list(set(data['user_id'].values))
    allUsers = structUser.Users(test_users, d, [])
    allUsers.user_parameters = allUsers.cal_user_cluster_parameters()
    arm_context = calArmFeatures.ArmFeatures()
    payoff = IjcaiCalReward.Payoff()
    user_id = test_users[:testUsersNum]
    BetaA = np.zeros((len(user_id), len(user_id)))
    BetaB = np.zeros((len(user_id), len(user_id)))
    for i in range(len(user_id)):
        for j in range(i, len(user_id)):
            if i == j:
                BetaA[i][j] = 2 * alpha
                BetaB[i][j] = 1
            else:
                BetaA[i][j] = alpha
                BetaA[j][i] = alpha
                BetaB[i][j] = alpha
                BetaB[j][i] = alpha
    # print(BetaB)
    # print(BetaA)
    # all the test user's behavior sequence
    sum_p = 0
    sum_r = 0
    sum_F1 = 0
    sum_reward = 0
    t = 0
    precisionList = []
    recallList = []
    F1List = []
    RewardList = []
    point = testUsersNum // 2
    for i in range(rounds):
        for j in range(testUsersNum):
            t += 1
            precision, recall, F1, reward = Neighbor(user_id, user_id[j], i,
                                                     gap)  # recommend for a single user in ith behavior history
            sum_p += precision
            sum_r += recall
            sum_F1 += F1
            sum_reward += reward
            if(t % point==0):
                avg_p = sum_p / t
                avg_r = sum_r / t
                avg_F1 = sum_F1 / t
                precisionList.append(avg_p)
                recallList.append(avg_r)
                F1List.append(avg_F1)
                RewardList.append(sum_reward)
    return precisionList, recallList, F1List, RewardList


if __name__ == '__main__':
    # read parameters from file into a dict
    fileName = "IjcaiParams.txt"
    fileObj = open(fileName, encoding='UTF-8')
    params = {}
    for line in fileObj:
        line = line.strip()
        key_value = line.split("=")
        if (len(key_value) == 2):
            params[key_value[0].strip()] = key_value[1]
    # convert the params into the desired type
    # set the params
    dimension = int(params['dimension'])
    dataset = str(params['dataset'])
    publicFile = "Params.txt"
    fileObj = open(publicFile, encoding='UTF-8')
    params = {}
    for line in fileObj:
        line = line.strip()
        key_value = line.split("=")
        if (len(key_value) == 2):
            params[key_value[0].strip()] = key_value[1]
    testUsersNum = int(params['testUsersNum'])
    rounds = int(params['rounds'])
    recomLength = int(params['recomLength'])
    armNum = int(params['armNum'])
    testIters = int(params['testIters'])
    alpha = int(params['alpha'])
    gap = float(params['gap'])
    name = str(params['name'])
    algorithm = str(params['algorithm'])
    precision = 0
    HR = 0
    F1 = 0
    reward = 0
    precisionArray = [[]] * testIters
    recallArray = [[]] * testIters
    F1Array = [[]] * testIters
    RewardArray = [[]] * testIters
    for i in tqdm(range(testIters)):
        p, r, F1, reward = main(testUsersNum, rounds, recomLength, dimension, armNum, gap, alpha)
        precisionArray[i] = p
        recallArray[i] = r
        F1Array[i] = F1
        RewardArray[i] = reward
    precisionArray = np.array(precisionArray)
    recallArray = np.array(recallArray)
    F1Array = np.array(F1Array)
    RewardArray = np.array(RewardArray)

    PrecisionRes = precisionArray.mean(axis=0)
    recallRes = recallArray.mean(axis=0)
    F1Res = F1Array.mean(axis=0)
    RewardRes = RewardArray.mean(axis=0)
    print("--------------------------Results-------------------------")
    print("precision：", PrecisionRes)
    print("Recall：", recallRes)
    print("F1", F1Res)
    print("reward", RewardRes)
    print("the number of test user", testUsersNum)
    print("the rounds of each user", rounds)
    print("iters", testIters)
    resFile = open('./res.txt', 'a')
    resFile.writelines(['\nprecision',dataset, " ", str(PrecisionRes)])
    resFile.writelines(['\nrecall', dataset, " ", str(recallRes)])
    resFile.writelines(['\nF1', dataset, " ", str(F1Res)])
    resFile.writelines(['\nreward', dataset, " ", str(RewardRes)])
    resFile.close()
    sendMail.send_email(name, dataset, algorithm)
