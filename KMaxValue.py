class kMaxValues:
    def kMaxValues(self, List, k):
        Lst = List[:]  # 对列表进行浅复制，避免后面更改原列表数据
        index_k = []
        for i in range(k):
            index_i = Lst.index(max(Lst))  # 得到列表的最d大值，并得到该最小值的索引
            index_k.append(index_i)  # 记录最小值索引
            Lst[index_i] = float('-inf')  # 将遍历过的列表最小值改为负无穷大，下次不再选择
        return index_k