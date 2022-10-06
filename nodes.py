


class InternalNode(object):
    def __init__(self, feature=0, value=0, child_L = None, child_R = None):
        self.feature = feature
        self.value = value
        self.child_L = child_L
        self.child_R = child_R
        self.name = 'feature ' + str(feature) + ' <= ' + str(value)
        self.leaf = False
        self.index = 0


class LeafNode(object):
    def __init__(self, action):
        self.action = action
        self.name = str(action)
        self.child_L = None
        self.child_R = None
        self.leaf = True
        self.index = 0