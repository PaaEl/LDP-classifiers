import math

import pandas as pd
from anytree import Node, RenderTree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from pure_ldp.frequency_oracles import LHClient, LHServer, DEClient, DEServer

class Tree(BaseEstimator,ClassifierMixin):
    def __init__(self, attrNames=None, depth=10, ldpMechanismClient=None, ldpMechanismServer=None,
                 epsilon_value=None, domainSize=None, max=None):
        if attrNames is not None:
            attrNames = [''.join(i for i in x if i.isalnum()).replace(' ', '_') for x in attrNames]
        self.attrNames = attrNames
        self.depth = depth
        self.ldpServer = ldpMechanismServer
        self.ldpClient = ldpMechanismClient
        self.root = None
        self.epsilon_value = epsilon_value
        self.domainSize = domainSize
        self.max = max
        self.nodes = {}

    def estimate(self,df, e, do):
        lis = []
        i = 0
        print(df)
        for x in df.columns:
            epsilon = e
            self.ldpServer.update_params(epsilon, do[i])
            self.ldpClient.update_params(epsilon, do[i])
            df.loc[:, x].apply(lambda g: self.ldpServer.aggregate(g))
            li = []
            for j in range(0, do[i]):
                li.append(round(self.ldpServer.estimate(j + 1)))
            lis.append(li)
            i += 1
        return lis

    def not_neg(lis):
        t = [[j if j > 0 else 0 for j in y] for y in lis]
        return t

    def rank(df, lis, c):
        ran = []
        i = 0
        for x in df.columns:
            # print(df.info)
            # print(len(lis))
            # print('lisi')
            # print(lis[i])
            s = Tree.gain(lis[i], c)
            tu = (s, x)
            ran.append(tu)
            i += 1
        return ran

    def entro(x):
        if x == 0:
            return 0
        else:
            return x * math.log2(x)

    def gain(lis, c):
        fraction = []
        prob = []
        b = sum(lis)
        if b == 0:
            b= 0.0000001
        i = 1
        j = 0
        while i < len(lis):
            bb = sum(lis[i - 1:i + c - 1])
            fraction.append(bb / b)
            while j < (i + c - 1):
                if bb == 0:
                    bbb = 0
                else:
                    bbb = lis[j] / bb
                prob.append(bbb)
                j += 1
            i += c
        enj = 0
        j = 0
        i = 1
        while i - 1 < len(fraction):
            en = 0
            while j < (i * c):
                en += Tree.entro(prob[j])
                j += 1
            enj += fraction[i - 1] * en
            i += 1
        return 1 + enj

    def create_node(feature, value, parent, count, le):
        return Node(feature + '#' + str(value), value = value, parent= parent,  count= [x * sum(count) / le for x in count])

    def grow_tree(self, parent,attrs_names, depth, run, do, amount, le):
        # print('dep')
        # print(depth)
        if parent is None:
            self.root = Node('root')
            self.nodes['root'] = self.root
            Tree.grow_tree(self, self.root, attrs_names, depth -1, run, do, amount,le)
        elif depth > 0:
            # if depth == 0:
            # print('elif')
            # print(self.nodes)
            # print(attrs_names)
            # print(do)
            run2 = [ii[0] for ii in run]
            o = run2.index(max(run2))
            sel = attrs_names[o]
            # print(category)
            # print(sel)
            sel2 = do[o]
            sel3 = amount[o]
            # print(sel2)
            # print('sel3')
            # print(sel3)
            i = 1
            j = 0
            while i <= sel2:
                sel4 = attrs_names[:o] + attrs_names[o+1:]
                # print('attr')
                # print(sel)
                # print(attrs_names)
                # print(sel4)
                sel5 = do[:o] + do[o+1:]
                sel6 = amount[:o] + amount[o+1:]
                sel7 = run[:o] + run[o+1:]
                # print('i')
                # print(i)
                lis = sel3[i-1:i+self.max-1]
                self.nodes[sel + '#'+ str(j)] = Tree.create_node(sel, j, parent, lis, le)
                # print(self.nodes)
                Tree.grow_tree(self, self.nodes[sel+ '#' + str(j)], sel4, depth - 1, sel7, sel5, sel6, le)
                j +=1
                i +=self.max
            # print(self.nodes)
        else:
            run2 = [ii[0] for ii in run]
            o = run2.index(max(run2))
            sel = attrs_names[o]
            # print(category)
            # print(sel)
            sel2 = do[o]
            sel3 = amount[o]
            # print(sel2)
            # print('sel3')
            # print(sel3)
            i = 1
            j = 0
            while i <= sel2:
                sel4 = attrs_names[:o] + attrs_names[o:]
                sel5 = do[:o] + do[o:]
                sel6 = amount[:o] + amount[o:]
                sel7 = run[:o] + run[o:]
                # print('i')
                # print(i)
                lis = sel3[i - 1:i + self.max - 1]
                self.nodes[sel + '#' + str(j)] = Tree.create_node(sel, j, parent, lis, le)
                j += 1
                i += self.max
            # print(self.nodes)
            return None

    def hash_perturb_get0(io):
        return io[0]

    def fit(self, X, y):
        # print('X')
        # print(X)
        perturbed_df_hash = pd.DataFrame()
        for x in X.columns:
            tempColumn = X.loc[:, x].apply(lambda item: Tree.hash_perturb_get0(item))
            perturbed_df_hash[x] = tempColumn
        T = X
        X, y = check_X_y(perturbed_df_hash, y)
        # X, y = check_X_y(X, y)
        self.X_ = X
        le = len(X)
        self.X_df_ = pd.DataFrame(X)
        self.y_ = y
        self.resultType = type(y[0])
        if self.attrNames is None:
            self.attrNames = [f'attr{x}' for x in range(len(self.X_[0]))]
        # print('ass')
        # print(self.attrNames)
        # print(self.X_[0])
        assert (len(self.attrNames) == len(self.X_[0]))

        data = [[] for i in range(len(self.attrNames))]
        categories = []

        for i in range(len(self.X_)):
            categories.append(str(self.y_[i]))
            for j in range(len(self.attrNames)):
                data[j].append(self.X_[i][j])
        w = Tree.estimate(self, T, self.epsilon_value, self.domainSize)
        # print('w')
        # print(w)
        n = Tree.not_neg(w)
        # print(n)
        run = Tree.rank(self.X_df_, n, self.max)
        # print('run')
        # print(run)
        # print(len(run))
        if self.depth > len(run):
            self.depth = len(run)

        self.tree_ = Tree.grow_tree(self, None,self.attrNames, self.depth, run, self.domainSize, n, le)
        # print(RenderTree(self.root))
        # print(self.root.children)
        # print('data')
        # print(data)
        # print(categories)

    def decision(root, obs, attrs_names, lis):
        if not root.children:
            return None
        else:
            # print(obs)
            # print(root.children[0])
            feat = root.children[0].name.split('#')[0]
            # print(feat)
            feat_ind = attrs_names.index(feat)
            # print(feat_ind)
            val = obs[feat_ind]
            # print(val)
            path = root.children[val]
            lis.append(path.count)
            # print(lis)
            Tree.decision(path, obs, attrs_names, lis)
            return lis



    def predict(self, X):
        check_is_fitted(self, ['tree_', 'resultType', 'attrNames'])
        # print('ccd')
        # print(X)
        X = check_array(X)
        # print(X)
        # print('wat')
        # print(type(X))
        prediction = []
        for i in range(len(X)):
            answer = Tree.decision(self.root,X[i],self.attrNames, [])
            # print('ans')
            # print(answer)
            g = [sum(i) for i in zip(*answer)]
            prediction.append(g)

        # print(prediction)
        g = [x.index(max(x)) for x in prediction]
        # print(g)
        return g
