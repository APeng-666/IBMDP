import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph
 
def plot_dtp(decision_tree_policy):

    dtp = Digraph(comment = 'Decision Tree Policy', node_attr ={'shape': 'box'})
    leaf_idx = -1
    internal_idx = 0
    # frontstack
    stack =[]
    node = decision_tree_policy
    node.index = internal_idx
    dtp.node(str(node.index), node.name)
    tmp = str(node.index)
    internal_idx +=1
    while node or stack:
        while node:
            stack.append(node)
            node = node.child_L
            if node:
                if node.leaf:
                    node.index = leaf_idx
                    dtp.node(str(node.index), node.name)
                    dtp.edge(tmp, str(node.index), color = "green")
                    leaf_idx -=1
                    tmp = str(leaf_idx)
                else:
                    node.index = internal_idx
                    dtp.node(str(node.index), node.name)
                    dtp.edge(tmp, str(node.index), color="green")
                    tmp = str(internal_idx)
                    internal_idx +=1                    
        node = stack.pop()
        tmp = str(node.index)
        node = node.child_R
        if node:
            if node.leaf:
                node.index = leaf_idx
                dtp.node(str(node.index), node.name)
                dtp.edge(tmp, str(node.index))
                leaf_idx -= 1
                tmp = str(leaf_idx)
            else:
                node.index = internal_idx
                dtp.node(str(node.index), node.name)
                dtp.edge(tmp, str(node.index))
                tmp = str(internal_idx)
                internal_idx +=1

    dtp.render('./models/dtp.gv', format='png', view=True)
    return
'''
def plot_losses(Q_loss_dtp, Q_loss_omi):
    plt.figure()
    plt.plot(np.array(Q_loss_dtp), color = 'r')
    plt.plot(np.array(Q_loss_omi), color = 'b')
    plt.legend(['QLoss_dtp', 'QLoss_omi'], loc='upper left')
    plt.savefig('./models/Q_losses.jpg')
    plt.close('all')
''' 

def plot_losses(Q_loss_dtp):
    plt.figure()
    plt.plot(np.array(Q_loss_dtp), color = 'r')
    plt.legend(['QLoss_dtp'], loc='upper left')
    plt.savefig('./models/Q_losses.jpg')
    plt.close('all') 