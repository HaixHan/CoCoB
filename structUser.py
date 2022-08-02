import numpy as np
import pandas as pd


class Users:
    def __init__(self, user_id, d, user_parameters):
        self.user_id = user_id
        self.d = d
        self.user_parameters = user_parameters


    def cal_user_cluster_parameters(self):
        user_id = self.user_id
        user_num = len(user_id)
        d = self.d
        m = np.identity(d)
        b = np.zeros((d, 1))
        list1 = [m for x in range(0, user_num)]
        list2 = [b for x in range(0, user_num)]
        user_parameters = pd.DataFrame(columns=('user_id', 'M', 'b'))
        user_parameters['user_id'] = user_id
        user_parameters['M'] = list1
        user_parameters['b'] = list2
        return user_parameters


    def find_user_parameters(self, u):
        user_parameters = self.user_parameters
        user_index = user_parameters[user_parameters['user_id'] == u].index[0]
        M = user_parameters.at[user_index, 'M']
        b = user_parameters.at[user_index, 'b']
        return M, b


    def caculate_cluster_parameters(self, users):
        d = self.d
        sum_M = np.identity(d)
        sum_b = np.zeros((d, 1))
        I = np.identity(d)
        user_size = len(users)
        for u in users:
            M, b = self.find_user_parameters(u)
            sum_M += (M - I)
            sum_b += b
        sum_w = np.linalg.inv(sum_M.T).dot(sum_b)
        return sum_M, sum_b, sum_w
