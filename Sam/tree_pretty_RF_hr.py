import copy
import math
from random import randrange

import pandas as pd
from anytree import Node, RenderTree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from pure_ldp.frequency_oracles import LHClient, LHServer, DEClient, DEServer, HadamardResponseServer, \
    HadamardResponseClient


class Tree(BaseEstimator,ClassifierMixin):
    def __init__(self, attrNames=None, depth=10, ldpMechanismClient=None, ldpMechanismServer=None,
                 epsilon_value=None, domainSize=None, max=None, tree=None):
        if attrNames is not None:
            attrNames = [''.join(i for i in x if i.isalnum()).replace(' ', '_') for x in attrNames]
        self.attr_names = attrNames
        self.depth = depth
        self.ldpServer = ldpMechanismServer
        self.ldpClient = ldpMechanismClient
        self.root = None
        self.epsilon_value = epsilon_value
        self.domainSize = domainSize
        self.max = max
        self.nodes = {}
        self.servers = []
        self.hfs = []
        self.tree = tree

    '''From pure ldp, perturbs the data'''

    def hash_perturb(io, client):
        g = client.privatise(io)
        return g

    def perturb(self, df, e, server, client, do):
        perturbed_df = pd.DataFrame()
        i = 0
        for x in df.columns:
            epsilon = e
            d = do[i]
            server = HadamardResponseServer(epsilon=epsilon, d=d)
            self.servers.append(server)
            hf = server.get_hash_funcs()
            self.hfs.append(hf)
            client = HadamardResponseClient(epsilon, d, hf)
            i += 1
            tempColumn = df.loc[:, x].apply(lambda item: Tree.hash_perturb(item + 1, client))
            perturbed_df[x] = tempColumn
        return perturbed_df

    '''Uses pure ldp module to estimate counts for each feature using frequency estimation'''

    def estimate(self, df, e, do, hff, hfs):
        lis = []
        i = 0
        for x in df.columns:
            epsilon = e
            self.ldpServer = hff[i]
            hf = hfs[i]
            self.ldpClient = HadamardResponseClient(epsilon, do[i], hf)
            df.loc[:, x].apply(lambda g: self.ldpServer.aggregate(g))
            self.ldpServer.estimate_all(range(1, do[i] + 1))
            f = list(self.ldpServer.estimated_data)
            lis.append(f)
            self.ldpServer.reset()
            i += 1
        return lis
    '''Negative counts set to 0'''
    def not_neg(lis):
        t = [[j if j > 0 else 0 for j in y] for y in lis]
        return t
    '''Labels each feature with it's information gain'''
    def rank(df, lis, c):
        ran = []
        i = 0
        for x in df.columns:
            s = Tree.gain(lis[i], c)
            tu = (s, x)
            ran.append(tu)
            i += 1
        return ran
    '''Used for calculating information gain'''
    def entro(x):
        if x == 0:
            return 0
        else:
            return x * math.log2(x)
    '''Calculates information gain'''
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
    '''Makes a node with feature name, value, parent and weight (count of feature value divided by total amount of records) '''

    def create_node(feature, value, parent, label, leaf):
        return Node(feature + '#' + str(value), value=value, parent=parent, label=label, is_leaf=leaf)

    def grow_tree(self, parent, attr_names, depth, feat_size, x, x_pert, hff, hfs):
        """

        @param parent: parent node
        @param attr_names: feature names
        @param depth: depth remaining
        @param run: list of features and their info gain
        @param feat_size: list of feature domainsizes
        @param amount: counts of feature values
        @param le: amount of records in total
        @return: tree
        """
        if parent is None:
            self.root = Node('root')
            self.nodes['root'] = self.root
            Tree.grow_tree(self, self.root, attr_names, depth, feat_size, x, x_pert, hff, hfs)
        elif depth > 1:
            o = randrange(len(attr_names))
            sel = attr_names[o]
            sel2 = feat_size[o]
            i = 1
            j = 0
            hf2 = copy.deepcopy(hff)
            del hf2[o]
            hf3 = copy.deepcopy(hfs)
            del hf3[o]
            while i <= sel2:
                sel4 = attr_names[:o] + attr_names[o + 1:]
                sel5 = feat_size[:o] + feat_size[o + 1:]
                orig_ind = x.index[x.iloc[:, o] == j].tolist()
                orig_df = x.take(orig_ind)
                orig_df=orig_df.drop(orig_df.columns[o], axis =1)
                orig_df= orig_df.reset_index(drop=True)
                pert_df = x_pert.take(orig_ind)
                pert_df=pert_df.drop(pert_df.columns[o], axis=1)
                pert_df=pert_df.reset_index(drop=True)
                self.nodes[sel + '#' + str(j)] = Tree.create_node(sel, j, parent, None, 0)
                Tree.grow_tree(self, self.nodes[sel + '#' + str(j)], sel4, depth - 1, sel5, orig_df, pert_df, hf2, hf3)
                j += 1
                i += self.max
        else:
            estimates = Tree.estimate(self, x_pert, self.epsilon_value, feat_size)
            pos_est = Tree.not_neg(estimates)
            o = randrange(len(attr_names))
            sel = attr_names[o]
            sel2 = feat_size[o]
            sel3 = pos_est[o]
            i = 1
            j = 0
            while i <= sel2:
                lis = sel3[i - 1:i + self.max - 1]
                g = lis.index(max(lis))
                self.nodes[sel + '#' + str(j)] = Tree.create_node(sel, j, parent, g, 1)
                j += 1
                i += self.max

            return None


    def fit(self, x,y, x_pert):
        """
        Fit data
        @param x: data
        @param y: labels
        """
        self.resultType = type(y[0])
        if self.depth > len(self.attr_names):
            self.depth = len(self.attr_names)

        self.tree_ = Tree.grow_tree(self, None,  self.attr_names,self.depth,self.domainSize, x, x_pert, self.tree.servers, self.tree.hfs)
        # print(RenderTree(self.root))
        # print(self.root.children)

    def decision(root, obs, attr_names):
        """
        Returns the predicted label for a record
        @param obs: the record
        @param attr_names: feature names
        @param lis: empty list
        @return: list of weights along the path to the leaf corresponding to the record
        """
        if not root.children:
            return None
        else:
            feat = root.children[0].name.split('#')[0]
            feat_ind = attr_names.index(feat)
            val = obs[feat_ind]
            path = root.children[val]
            if path.is_leaf == 1:
                return path.label
            else:
                return Tree.decision(path, obs, attr_names)



    def predict(self, X):
        """
        Predict the label for a record by adding the weights of all possible labels and selecting the max one
        @param X: record
        @return: label
        """
        X = check_array(X)
        prediction = []
        for i in range(len(X)):
            answer = Tree.decision(self.root, X[i], self.attr_names)
            prediction.append(answer)
        return prediction
