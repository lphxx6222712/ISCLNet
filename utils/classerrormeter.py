import numpy as np
import torch
import numbers
from torchnet.meter import meter
from sklearn import metrics as mt
import matplotlib.pyplot as plt

class aucmeter(meter.Meter):
    """
    The AUCMeter measures the area under the receiver-operating characteristic
    (ROC) curve for binary classification problems. The area under the curve (AUC)
    can be interpreted as the probability that, given a randomly selected positive
    example and a randomly selected negative example, the positive example is
    assigned a higher score by the classification model than the negative example.

    The AUCMeter is designed to operate on one-dimensional Tensors `output`
    and `target`, where (1) the `output` contains model output scores that ought to
    be higher when the model is more convinced that the example should be positively
    labeled, and smaller when the model believes the example should be negatively
    labeled (for instance, the output of a signoid function); and (2) the `target`
    contains only values 0 (for negative examples) and 1 (for positive examples).
    """

    def __init__(self):
        super(aucmeter, self).__init__()
        self.reset()

    def draw_roc(slef,fpr,tpr, auc):
        plt.figure()
        lw = 2
        plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve (area = %0.2f)'%auc)
        plt.plot([0,1],[0,1],color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristive example')
        plt.legend(loc = "lower right")
        plt.show()

    def reset(self):
        self.scores = torch.DoubleTensor(torch.DoubleStorage()).numpy()
        self.targets = torch.LongTensor(torch.LongStorage()).numpy()

    def add(self, output, target):
        # if torch.is_tensor(output):
        #     output = output.cpu().squeeze().numpy()
        # if torch.is_tensor(target):
        #     target = target.cpu().squeeze().numpy()
        # elif isinstance(target, numbers.Number):
        #     target = np.asarray([target])
        # #assert np.ndim(output) == 1, \
        #    'wrong output size (1D expected)'
        # assert np.ndim(target) == 1, \
        #    'wrong target size (1D expected)'
        # assert output.shape[0] == target.shape[0], \
        #   'number of outputs and targets does not match'
        assert np.all(np.add(np.equal(target, 1), np.equal(target, 0))), \
            'targets should be binary (0, 1)'
        if len(self.scores) == 0:
            self.scores = output
            self.targets = target
        else:
            self.scores = np.append(self.scores, output, axis=0)
            # self.scores = np.stack(self.scores, output, axis=1)
            self.targets = np.append(self.targets, target, axis=0)

    def value(self):
        # case when number of elements added are 0
        if self.scores.shape[0] == 0:
            return 0.5
        accuracy, precision, recall, f1_score, area, tpr, fpr = [], [], [], [], [], [], []
        prediction = self.scores > 0.5
        prediction = prediction.astype('uint8')
        for i in range(self.targets.shape[1]):
            accuracy1 = mt.balanced_accuracy_score(self.targets[:,i], prediction[:,i])
            accuracy.append(accuracy1)
            precision1, recall1, f1_score1, supports1 = mt.precision_recall_fscore_support(self.targets[:,i], prediction[:,i])
            area.append(mt.roc_auc_score(self.targets[:,i], self.scores[:,i]))
            precision.append(precision1)
            recall.append(recall1)
            f1_score.append(f1_score1)

        # sorting the arrays
        self.scores = self.scores.reshape(-1)
        self.targets = self.targets.reshape(-1)
        scores, sortind = torch.sort(torch.from_numpy(
            self.scores), dim=0, descending=True)
        scores = scores.numpy()
        sortind = sortind.numpy()

        # creating the roc curve
        tpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)
        fpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)

        for i in range(1, scores.size + 1):
            if self.targets[sortind[i - 1]] == 1:
                tpr[i] = tpr[i - 1] + 1
                fpr[i] = fpr[i - 1]
            else:
                tpr[i] = tpr[i - 1]
                fpr[i] = fpr[i - 1] + 1

        tpr /= (self.targets.sum() * 1.0)
        fpr /= ((self.targets - 1.0).sum() * -1.0)

        # calculating area under curve using trapezoidal rule
        n = tpr.shape[0]
        h = fpr[1:n] - fpr[0:n - 1]
        sum_h = np.zeros(fpr.shape)
        sum_h[0:n - 1] = h
        sum_h[1:n] += h
        #area = (sum_h * tpr).sum() / 2.0
        area = np.mean(area)
        accuracy = np.mean(accuracy)
        precision = np.mean(precision)
        recall = np.mean(recall)
        f1_score = np.mean(f1_score)

        #self.draw_roc(fpr,tpr,area)
        return accuracy, precision, recall, f1_score, area, tpr, fpr
