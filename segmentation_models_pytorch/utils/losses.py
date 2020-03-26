import torch.nn as nn

from . import base
from . import functional as F
from  .base import Activation


class JaccardLoss(base.Loss):

    def __init__(self, eps=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        if isinstance(y_pr, tuple):
            y_pr = y_pr[0]
        
        jaccard_score = F.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )

        assert jaccard_score >= 0, 'Theres something wrong with your jaccard loss'

        return 1 - jaccard_score


class DiceLoss(base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        if isinstance(y_pr, tuple):
            y_pr = y_pr[0]
        
        #print('-', y_pr.size() ,'-')
        f1_score = F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )

        assert f1_score >= 0, 'Theres something wrong with your dice loss'

        return 1 - f1_score


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass
    #def __init__(self, eps=1e-4, weight=None, activation=None, ignore_channels=None, **kwargs):
        #super().__init__(**kwargs)
        #self.eps = eps
        #self.weight = weight
        #self.activation = Activation(activation)
        #self.ignore_channels = ignore_channels

    #def forward(self, y_pr, y_gt):
        #y_pr = self.activation(y_pr[0])
        #cross_entropy_loss = nn.functional.cross_entropy(y_pr, y_gt, self.weight)
        #return torch.mean(cross_entropy_loss)

class CategoricalFocalLoss(base.Loss):
    pass
    #def __init__(self, eps=1e-4, gamma=2.0, alpha=0.25, class_indexes=None, activation=None, ignore_channels=None, **kwargs):
        #super().__init__(**kwargs)
        #self.eps = eps
        #self.gamma = gamma
        #self.alpha = alpha
        #self.class_indexes = class_indexes
        #self.weight = weight
        #self.activation = Activation(activation)
        #self.ignore_channels = ignore_channels

    #def forward(self, y_pr, y_gt):
        #y_pr = self.activation(y_pr[0])
        #categorical_focal_loss = F.categorical_focal_loss(y_pr, y_gt, self.weight)
        #return torch.mean(categorical_focal_loss)

class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(base.Loss):
    def __init__(self, eps=1e-4, weight=None, activation=None, threshold=0.5, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.weight = weight
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        if isinstance(y_pr, tuple):
            y_pr = y_pr[0]

        #print('*', y_pr.size(), '*')
        #print('*', y_gt.size(), '*')
        y_pr = F._threshold(y_pr, threshold=self.threshold)
        y_pr, y_gt = F._take_channels(y_pr, y_gt, ignore_channels=self.ignore_channels)

        #print('*', y_pr.size() ,'*')
        #print('*', y_gt.size() ,'*')
        bce_loss = nn.functional.binary_cross_entropy(y_pr, y_gt, self.weight)
        return bce_loss


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass
    #def __init__(self, eps=1e-4, weight=None, activation=None, ignore_channels=None, **kwargs):
        #super().__init__(**kwargs)
        #self.eps = eps
        #self.weight = weight
        #self.activation = Activation(activation)
        #self.ignore_channels = ignore_channels

    #def forward(self, y_pr, y_gt):
        #y_pr = self.activation(y_pr[0])
        #bce_w_logit_loss = nn.functional.binary_cross_entropy_with_logits(y_pr, y_gt, self.weight)
        #return torch.mean(bce_w_logit_loss)