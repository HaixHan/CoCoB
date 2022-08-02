class kMaxValues:
    def kMaxValues(self, List, k):
        Lst = List[:]  
        index_k = []
        for i in range(k):
            index_i = Lst.index(max(Lst))  
            index_k.append(index_i) 
            Lst[index_i] = float('-inf')  
        return index_k
