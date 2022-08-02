import numpy as np
import pandas as pd



class ArmFeatures:
    def find_item_feature(self, item, item_features, d):
        position = list(item_features['item_id'].values)
        index = position.index(item)
        feature = np.array(item_features.iloc[index].values[1:]).reshape((d, -1))
        return feature

    def cal_arm_features(self, item_list, item_features, d):
        x = np.zeros((d, 1))
        length = len(item_list)
        if (length > 0):
            position = list(item_features['item_id'].values)
            for item_id in item_list:
                index = position.index(item_id)
                feature = np.array(item_features.iloc[index].values[1:]).reshape((d, -1))
                x += feature
            x = x / length
        return x
