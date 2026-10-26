import torch
from torch import nn
from torch.nn import functional as F
from mmseg.registry import MODELS

@MODELS.register_module()
class PixelMaskedDiceLoss(nn.Module):
    def __init__(self,use_sigmoid=True,activate=True,reduction='mean',
                 naive_dice=False,loss_weight=1.0,ignore_index=255,
                 eps=1e-3,loss_name='loss_dice'):
        super().__init__()
        if reduction not in ('none','mean','sum'):raise ValueError(reduction)
        if eps<=0:raise ValueError('eps must be positive')
        self.use_sigmoid=use_sigmoid;self.activate=activate
        self.reduction=reduction;self.naive_dice=naive_dice
        self.loss_weight=loss_weight;self.ignore_index=ignore_index
        self.eps=eps;self._loss_name=loss_name

    @property
    def loss_name(self):return self._loss_name

    def forward(self,pred,target,weight=None,avg_factor=None,
                reduction_override=None,ignore_index=None,**kwargs):
        if target.ndim==pred.ndim and target.shape[1]==1:target=target.squeeze(1)
        if pred.ndim!=4 or target.ndim!=3:raise ValueError('Expected NCHW logits and NHW labels')
        index=self.ignore_index if ignore_index is None else ignore_index
        valid=target!=index
        safe=target.masked_fill(~valid,0).long()
        truth=F.one_hot(safe,num_classes=pred.shape[1]).permute(0,3,1,2).float()

        probs=pred.float()
        if self.activate:
            probs=probs.sigmoid() if self.use_sigmoid else probs.softmax(dim=1)
        mask=valid.unsqueeze(1)
        probs=probs*mask;truth=truth*mask
        a=(probs*truth).flatten(1).sum(1)
        if self.naive_dice:
            denominator=probs.flatten(1).sum(1)+truth.flatten(1).sum(1)+self.eps
            overlap=(2*a+self.eps)/denominator
        else:
            denominator=probs.square().flatten(1).sum(1)+truth.square().flatten(1).sum(1)+2*self.eps
            overlap=2*a/denominator
        losses=torch.where(valid.flatten(1).any(1),1-overlap,torch.zeros_like(overlap))
        if weight is not None:
            if weight.shape!=losses.shape:raise ValueError('Dice weight must contain one value per image')
            losses=losses*weight
        reduction=reduction_override or self.reduction
        if reduction=='mean':
            losses=losses.mean() if avg_factor is None else losses.sum()/avg_factor
        elif reduction=='sum':
            if avg_factor is not None:raise ValueError('avg_factor cannot be used with sum')
            losses=losses.sum()
        elif reduction!='none':raise ValueError(reduction)
        return self.loss_weight*losses
