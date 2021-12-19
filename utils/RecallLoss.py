import torch
import torch.nn.functional as F

memory = None


class RecallCrossEntropy(torch.nn.Module):
    def __init__(self, n_classes=7, ignore_index=-1, weight_memory=None, weight_rate=0.6):
        super(RecallCrossEntropy, self).__init__()
        self.n_classes = n_classes + 1
        self.ignore_index = ignore_index
        self.weight_memory = weight_memory
        self.weight_rate = weight_rate
        # 增加计算pusead lable 标签 noise
        self.log_sm = torch.nn.LogSoftmax(dim=1)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, input, target):
        torch.cuda.empty_cache()
        # input (batch,n_classes,H,W)
        # target (batch,H,W)
        pred = input.argmax(1)
        idex = (pred != target).view(-1)

        # calculate ground truth counts
        gt_counter = torch.ones((self.n_classes,)).cuda()
        gt_idx, gt_count = torch.unique(target, return_counts=True)

        # map ignored label to an exisiting one
        gt_count[gt_idx == self.ignore_index] = gt_count[0].clone()
        gt_idx[gt_idx == self.ignore_index] = 7
        gt_counter[gt_idx] = gt_count.float()

        # calculate false negative counts
        fn_counter = torch.ones((self.n_classes)).cuda()
        fn = target.view(-1)[idex]
        # INUQUE

        fn_idx, fn_count = torch.unique(fn, return_counts=True)

        # map ignored label to an exisiting one
        fn_count[fn_idx == self.ignore_index] = fn_count[0].clone()  # error out of bound.
        fn_idx[fn_idx == self.ignore_index] = 7
        fn_counter[fn_idx] = fn_count.float()

        weight = fn_counter / gt_counter
        if self.weight_memory is None:
            self.weight_memory = weight
        else:
            weight = self.weight_rate * weight + self.weight_memory * (1 - self.weight_rate)
            self.weight_memory = weight
        CE = F.cross_entropy(input, target, reduction='mean', ignore_index=self.ignore_index,
                             weight=weight[:self.n_classes - 1])
        return CE

class CEWithPseudo(torch.nn.Module):
    def __init__(self, ignore_index=-1):
        super(CEWithPseudo, self).__init__()
        self.ignore_index = ignore_index
        # 增加计算pusead lable 标签 noise
        self.log_sm = torch.nn.LogSoftmax(dim=1)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, pred1, pred2, target):
        CE = F.cross_entropy(pred1, target, reduction='none', ignore_index=self.ignore_index)
        # calculate pseudo
        variance = torch.sum(F.kl_div(self.log_sm(pred1), self.sm(pred2),reduction='none'), dim=1)
        exp_variance = torch.exp(-variance)
        loss = torch.mean(CE * exp_variance) + torch.mean(variance)
        return loss


class RecallCrossEntropyWithPseudo(torch.nn.Module):
    def __init__(self, n_classes=7, ignore_index=-1, weight_memory=None, weight_rate=0.6):
        super(RecallCrossEntropyWithPseudo, self).__init__()
        self.n_classes = n_classes + 1
        self.ignore_index = ignore_index
        self.weight_memory = weight_memory
        self.weight_rate = weight_rate
        # 增加计算pusead lable 标签 noise
        self.log_sm = torch.nn.LogSoftmax(dim=1)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, pred1, pred2, target):
        """
        pred1 是主分類器
        pred2 是輔助分類器
        """
        # input (batch,n_classes,H,W)
        # target (batch,H,W)
        pred = pred1.argmax(1)
        idex = (pred != target).view(-1)

        # calculate ground truth counts
        gt_counter = torch.ones((self.n_classes,)).cuda()
        gt_idx, gt_count = torch.unique(target, return_counts=True)

        # map ignored label to an exisiting one
        gt_count[gt_idx == self.ignore_index] = gt_count[0].clone()
        gt_idx[gt_idx == self.ignore_index] = 7
        gt_counter[gt_idx] = gt_count.float()

        # calculate false negative counts
        fn_counter = torch.ones((self.n_classes)).cuda()
        fn = target.view(-1)[idex]

        fn_idx, fn_count = torch.unique(fn, return_counts=True)

        # map ignored label to an exisiting one
        fn_count[fn_idx == self.ignore_index] = fn_count[0].clone()  # error out of bound.
        fn_idx[fn_idx == self.ignore_index] = 7
        fn_counter[fn_idx] = fn_count.float()

        weight = fn_counter / gt_counter
        if self.weight_memory is None:
            self.weight_memory = weight
        else:
            weight = self.weight_rate * weight + self.weight_memory * (1 - self.weight_rate)
            self.weight_memory = weight
        CE = F.cross_entropy(pred1, target, reduction='none', ignore_index=self.ignore_index,
                             weight=weight[:self.n_classes - 1])

        # calculate pseudo
        variance = torch.sum(F.kl_div(self.log_sm(pred1), self.sm(pred2),reduction='none'), dim=1)
        exp_variance = torch.exp(-variance)
        loss = torch.mean(CE * exp_variance) + torch.mean(variance)
        return loss


if __name__ == '__main__':
    pass
