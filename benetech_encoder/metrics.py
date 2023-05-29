from torchmetrics import Metric
import torch

import re
import numpy as np
from rapidfuzz.distance.Levenshtein import distance as levenshtein
from sklearn.metrics import r2_score

class BenetechMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, trues: str, preds: str):
        scores = []
        for gt_series, pred_series in zip(trues, preds):
            
            gt_processed = self.process_prediction(gt_series)
            pred_processed = self.process_prediction(pred_series)
            
            for gt_arr, pred_arr in zip(gt_processed, pred_processed):
                
                # Sanity Check: Make sure true labels can be parsed
                assert gt_arr != None

                # If string conversion failed return 0
                if pred_arr == None:
                    scores.append(0)

                # If GT is string, convert predicted series to strings
                if isinstance(gt_arr[0], str):
                    pred_arr = [str(x) for x in pred_arr]

                # Return 0 if pred_series is str and ground truth is not
                if isinstance(gt_arr[0], str) == False and isinstance(pred_arr[0], str) == True:
                    scores.append(0)
                    
                # Score with RMSE or Levenshtein as appropriate
                else:
                    scores.append(self.score_series(gt_arr, pred_arr))
                    
        self.score += torch.sum(torch.tensor(scores, dtype=torch.double), dtype=torch.double)
        self.total += torch.tensor(len(scores), dtype=torch.double)

    def compute(self):
        return self.score.float() / self.total
  
    def normalized_rmse(self, y_true, y_pred):
        # The argument to the sigmoid transform is equal to 
        # rmse(y_true, y_pred) / rmse(y_true, np.mean(y_true))
        return self.sigmoid((1 - np.clip(r2_score(y_true, y_pred), 0, 1) ** 0.5))
    
    def sigmoid(self, x):
        return 2 - 2 / (1 + np.exp(-x))
    
    def normalized_levenshtein_score(self, y_true, y_pred):
        total_distance = np.sum([levenshtein(yt, yp) for yt, yp in zip(y_true, y_pred)])
        length_sum = np.sum([len(yt) for yt in y_true])
        return self.sigmoid(total_distance / length_sum)

    def score_series(self, y_true, y_pred):
        if len(y_true) != len(y_pred):
            return 0.0
        if isinstance(y_true[0], str):
            return self.normalized_levenshtein_score(y_true, y_pred)
        else:
            return self.normalized_rmse(y_true, y_pred)
    
    def process_prediction(self, string):
        try:
            tmp = tmp = [x.strip().split(" ") for x in re.sub(r"<0x0A>$", "", string.strip()).split("<0x0A>")]
            xs = [x[0] for x in tmp]
            ys = [x[-1] for x in tmp]

            xs = self.convert_arr(xs)
            ys = self.convert_arr(ys)
            return [xs, ys]
        except:
            return [None, None]
        
    def convert_arr(self, arr):
        """
        Helper to get the type of series.

        strings: 2
        floats: 1
        ints: 0
        """
        floats = False
        ints = True

        # iterate values in arr (break if string)
        # TODO: IMPROVE THE WAY I preprocess HERE (look at results of predictions to determine)
        for val in arr:
            if not re.search('^[0-9.-]*$', val) or len(val) == 0 or \
                  (val.count('.') > 1 or val[-1] == '.' or val[0] == '.') or \
                  (val.count('-') == 1 and val[0] != "-") or (val.count('-') >= 2) or (len(val) == 1 and val[0] not in "123456789"):
                return arr
            elif floats == False and '.' in val:
                floats = True

        if floats == True:
            return [float(x) for x in arr]
        else:
            return [int(x) for x in arr]