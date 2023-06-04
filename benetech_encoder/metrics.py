from torchmetrics import Metric
import torch

import re, math
import numpy as np
from rapidfuzz.distance.Levenshtein import distance as levenshtein
from sklearn.metrics import r2_score

from torchmetrics import Metric

class BenetechMetric(Metric):
    def __init__(self, chart_type, axis):
        super().__init__()
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.chart_type = chart_type
        self.is_x_values = self.check_axis(axis)

    def check_axis(self, axis):
        if axis == "x":
            return True
        elif axis == "y":
            return False
        else:
            raise ValueError(f"{axis} is not a recognized axis")

    def update(self, trues: list, preds: list, testing: bool = False):
        scores = []
        for gt_series, pred_series in zip(trues, preds):
            if testing == True:
                gt_processed = [gt_series.split(";")]
                pred_processed = [pred_series.split(";")]
            else:
                gt_processed = self.process_prediction(gt_series)
                pred_processed = self.process_prediction(pred_series)                

            for i, (true_arr, pred_arr) in enumerate(zip(gt_processed, pred_processed)):

                # Sanity Check: Make sure true labels can be parsed
                assert true_arr != None

                # If string conversion failed return 0
                if pred_arr == None:
                    scores.append(0)
                    continue

                # Graph conventions
                if self.chart_type == "v":
                    if self.is_x_values == False:
                        if self.check_float_conversion(pred_arr) == False:
                            scores.append(0.0)
                            continue
                        else:
                            pred_arr = [float(val) for val in pred_arr]
                            true_arr = [float(val) for val in true_arr]
                    
                elif self.chart_type == "h":
                    if self.is_x_values == True:
                        if self.check_float_conversion(pred_arr) == False:
                            scores.append(0.0)
                            continue
                        else:
                            pred_arr = [float(val) for val in pred_arr]
                            true_arr = [float(val) for val in true_arr]
                    
                
                elif self.chart_type == "s":
                    if self.is_x_values in [True, False]:
                        if self.check_float_conversion(pred_arr) == False:
                            scores.append(0.0)
                            continue
                        else:
                            pred_arr = [float(val) for val in pred_arr]
                            true_arr = [float(val) for val in true_arr]
                    
                elif self.chart_type == "l":
                    if self.is_x_values == False:
                        if self.check_float_conversion(pred_arr) == False:
                            scores.append(0.0)
                            continue
                        else:
                            pred_arr = [float(val) for val in pred_arr]
                            true_arr = [float(val) for val in true_arr]
                    
                elif self.chart_type == "d":
                    if self.is_x_values == False:
                        if self.check_float_conversion(pred_arr) == False:
                            scores.append(0.0)
                            continue
                        else:
                            pred_arr = [float(val) for val in pred_arr]
                            true_arr = [float(val) for val in true_arr]
                else:
                    raise ValueError(f"{self.chart_type} not a recognized chart type.")

                if len(true_arr) != len(pred_arr):
                    score = 0.0
                elif isinstance(true_arr[0], str):
                    score = self.normalized_levenshtein_score(true_arr, pred_arr)
                else:
                    # Sanity check
                    true_arr = self.replace_nan(true_arr)
                    pred_arr = self.replace_nan(pred_arr)
                    try:
                        score = self.normalized_rmse(true_arr, pred_arr)
                    except:
                        score = 0.0
                scores.append(score)

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
        arr = [x.strip() for x in re.sub(r"<0x0A>$", "", string.strip()).split("<0x0A>")]
        return [arr]
        
    def check_float_conversion(self, arr):
        try:
            for element in arr:
                float(element)
        except ValueError:
            return False
        return True
            
    def convert_arr(self, arr):
        """
        Helper to get the type of series.

        strings: 2
        floats: 1
        ints: 0s
        """
        if self.check_float_conversion(arr):
            arr = [float(x) for x in arr]
            return self.replace_nan(arr)
        elif self.check_float_conversion(arr[1:]):
            arr = [float(arr[1])] + [float(x) for x in arr[1:]]
            return self.replace_nan(arr)
        return arr

    def replace_nan(self, arr):
        res = []
        for num in arr:
            if math.isnan(num) or math.isinf(num):
                res.append(0.0)
            else:
                res.append(num)
        return res