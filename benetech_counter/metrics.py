from torchmetrics import Metric
import torch

class CounterAccuracy(Metric):
    def __init__(self, ):
        super().__init__()
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, trues: list, preds: list):
        scores = []
        for gt, pred in zip(trues, preds):
            # if gt == pred:
            #     scores.append(1)
            if gt.count("<0x0A>") == pred.count("<0x0A>"):
                scores.append(1)
            else:
                scores.append(0)

        self.score += torch.sum(torch.tensor(scores, dtype=torch.double), dtype=torch.double)
        self.total += torch.tensor(len(scores), dtype=torch.double)

    def compute(self):
        return self.score.float() / self.total