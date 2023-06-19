from torchmetrics import Metric
import torch

import re
import numpy as np
from rapidfuzz.distance.Levenshtein import distance as levenshtein
from sklearn.metrics import r2_score

"""
Benetech competition metric from: https://www.kaggle.com/code/ryanholbrook/competition-metric-benetech-mixed-match
- Converted to a torchmetrics object for usage with torch lightning
"""

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
            tmp = [x.strip().split(" ") for x in string.split("|")]
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

if __name__ == "__main__":

    # Simple example to show how the metric works
    gts = ['0 <0x0A> 0.01 | 6 <0x0A> -0.68 | 12 <0x0A> -1.38 | 18 <0x0A> -2.08 | 24 <0x0A> -2.74', '0.0 <0x0A> 0.013278 | 0.4 <0x0A> 0.0132781 | 0.8 <0x0A> 0.0132824 | 1.2 <0x0A> 0.0132724 | 1.6 <0x0A> 0.0132788 | 2.0 <0x0A> 0.0132701 | 2.4 <0x0A> 0.013278', 'Group 1 <0x0A> 3.7 | Group 2 <0x0A> 8.4', '5 <0x0A> 11 | 5 <0x0A> 12 | 5 <0x0A> 14 | 6 <0x0A> 12 | 6 <0x0A> 13 | 6 <0x0A> 14 | 7 <0x0A> 14 | 7 <0x0A> 16 | 7 <0x0A> 17 | 8 <0x0A> 17 | 8 <0x0A> 18 | 8 <0x0A> 19 | 9 <0x0A> 20 | 9 <0x0A> 21 | 9 <0x0A> 22 | 10 <0x0A> 21 | 10 <0x0A> 22 | 10 <0x0A> 23 | 11 <0x0A> 21 | 11 <0x0A> 22 | 11 <0x0A> 23 | 11 <0x0A> 24 | 12 <0x0A> 23 | 12 <0x0A> 24 | 12 <0x0A> 25 | 12 <0x0A> 26 | 13 <0x0A> 24 | 13 <0x0A> 26 | 13 <0x0A> 27 | 14 <0x0A> 25 | 14 <0x0A> 27 | 14 <0x0A> 28 | 15 <0x0A> 26 | 15 <0x0A> 27 | 15 <0x0A> 29 | 16 <0x0A> 29 | 16 <0x0A> 30 | 16 <0x0A> 31', '21-Feb <0x0A> 89556 | 22-Feb <0x0A> 150266 | 23-Feb <0x0A> 170845 | 24-Feb <0x0A> 175990 | 25-Feb <0x0A> 136889 | 26-Feb <0x0A> 99846 | 27-Feb <0x0A> 1063 | 28-Feb <0x0A> 42223 | 29-Feb <0x0A> 59715 | 01-Mar <0x0A> 66918 | 02-Mar <0x0A> 54570 | 03-Mar <0x0A> 44281 | 04-Mar <0x0A> 64860 | 05-Mar <0x0A> 80295 | 06-Mar <0x0A> 82353 | 07-Mar <0x0A> 102933 | 08-Mar <0x0A> 129686 | 09-Mar <0x0A> 101904 | 10-Mar <0x0A> 9295', '0 <0x0A> 0.0001 | 50 <0x0A> 0.00397 | 100 <0x0A> 0.00369 | 150 <0x0A> 0.00306 | 200 <0x0A> 0.00246 | 250 <0x0A> 0.00194 | 300 <0x0A> 0.00148 | 350 <0x0A> 0.00108 | 400 <0x0A> 0.00073 | 450 <0x0A> 0.00046 | 500 <0x0A> 0.0002', '1995 <0x0A> 50763 | 1996 <0x0A> 41876 | 1997 <0x0A> 26945 | 1998 <0x0A> 33521 | 1999 <0x0A> 32887 | 2000 <0x0A> 27174 | 2001 <0x0A> 28320 | 2002* <0x0A> 23050 | 2003* <0x0A> 21217 | 2004* <0x0A> 21446', 'Jan-01 <0x0A> 1008 | Apr-01 <0x0A> 982 | Jul-01 <0x0A> 967 | Oct-01 <0x0A> 956 | Jan-02 <0x0A> 961 | Apr-02 <0x0A> 953 | Jul-02 <0x0A> 956 | Oct-02 <0x0A> 949', 'Rest <0x0A> 0.035 | Water Placebo <0x0A> 0.049 | 4% Glucose beverage <0x0A> 0.053 | No fluid <0x0A> 0.063', '1.0 <0x0A> 15.0 | 2.1 <0x0A> 14.5 | 3.1 <0x0A> 11.0 | 4.1 <0x0A> 11.5 | 5.0 <0x0A> 11.0 | 6.0 <0x0A> 11.0 | 7.0 <0x0A> 10.5 | 8.0 <0x0A> 10.5 | 9.0 <0x0A> 10.0 | 10.0 <0x0A> 9.4 | 10.9 <0x0A> 9.5 | 12.0 <0x0A> 9.0 | 12.9 <0x0A> 9.0 | 13.9 <0x0A> 9.0 | 14.8 <0x0A> 9.1 | 15.9 <0x0A> 9.3 | 16.9 <0x0A> 9.3 | 17.8 <0x0A> 9.3 | 18.8 <0x0A> 8.0 | 19.8 <0x0A> 8.0']
    pts = ['0 <0x0A> 0.0 | 6 <0x0A> -1.3 | 12 <0x0A> -2.0 | 18 <0x0A> -1.9 | 24 <0x0A> -3.2', '0 <0x0A> 0.013284 | 0.4 <0x0A> 0.013269 | 0.8 <0x0A> 0.013284 | 1.2 <0x0A> 0.013269 | 1.6 <0x0A> 0.013277 | 2.0 <0x0A> 0.013268 | 2.4 <0x0A> 0.013277 | 2.8 <0x0A> 0.013284', 'Group 1 <0x0A> 3.6 | Group 2 <0x0A> 8.4', '1.0 <0x0A> 14.7 | 2.0 <0x0A> 14.5 | 3.0 <0x0A> 14.6 | 4.0 <0x0A> 14.8 | 5.0 <0x0A> 14.8 | 6.0 <0x0A> 14.8 | 7.0 <0x0A> 14.8 | 8.0 <0x0A> 14.9 | 9.0 <0x0A> 14.6 | 10.0 <0x0A> 14.6 | 11.0 <0x0A> 14.9 | 12.0 <0x0A> 14.9 | 13.0 <0x0A> 14.9 | 14.0 <0x0A> 14.9 | 15.0 <0x0A> 14.9 | 16.0 <0x0A> 14.9 | 17.0 <0x0A> 24.4 | 18.0 <0x0A> 24.4 | 19.0 <0x0A> 24.4 | 20.0 <0x0A> 24.0 | 21.0 <0x0A> 24.0 | 22.0 <0x0A> 24.0 | 23.0 <0x0A> 24.0 | 24.0 <0x0A> 24.0 | 25.0 <0x0A> 24.0 | 26.0 <0x0A> 24.0 | 27.0 <0x0A> 24.0 | 28.0 <0x0A> 24.0 | 29.0 <0x0A> 24.0 | 29.9 <0x0A> 25.0 | 29.9 <0x0A> 24.0 | 30.0 <0x0A> 24.0 | 31.0 <0x0A> 24.0 | 32.0 <0x0A> 24.0 | 33.0 <0x0A> 24.0 | 34.0 <0x0A> 24.0 | 35.0 <0x0A> 24.0 | 36.0 <0x0A> 24.0 | 37.0 <0x0A> 24.0 | 38.0 <0x0A> 24.0', '21-268 <0x0A> 88832 | 22-289 <0x0A> 150523 | 23-299 <0x0A> 171393 | 24-299 <0x0A> 175999 | 25-299 <0x0A> 136959 | 268-299 <0x0A> 98499 | 270-28 <0x0A> 40220 | 2748 <0x0A> 60220 | 2778 <0x0A> 60220 | 2849 <0x0A> 65999 | 2924 <0x0A> 54220 | 2240 <0x0A> 54220 | 2244 <0x0A> 42899 | 2254 <0x0A> 64299 | 2268 <0x0A> 80220 | 2264 <0x0A> 102200 | 2268 <0x0A> 102200 | 2269 <0x0A> 102200 | 2304 <0x0A> 102200', '0 <0x0A> 0.00000 | 50 <0x0A> 0.00000 | 100 <0x0A> 0.00033 | 150 <0x0A> 0.00315 | 200 <0x0A> 0.00241 | 250 <0x0A> 0.00168 | 300 <0x0A> 0.00155 | 350 <0x0A> 0.000085 | 400 <0x0A> 0.00077 | 450 <0x0A> 0.00011 | 500 <0x0A> 0.00005', '1995 <0x0A> 51222.4 | 1996 <0x0A> 41882.4 | 1997 <0x0A> 26979.4 | 1998 <0x0A> 33341.4 | 1999 <0x0A> 32709.4 | 2000 <0x0A> 27199.4 | 2001 <0x0A> 28199.4 | 2002* <0x0A> 23099.4 | 2003* <0x0A> 21419.4 | 2004* <0x0A> 21719.4', 'Jan-01 <0x0A> 1006.8 | Apr-01 <0x0A> 980.4 | Jul-01 <0x0A> 963.9 | Oct-01 <0x0A> 955.4 | Jan-02 <0x0A> 961.9 | Apr-02 <0x0A> 951.9 | Jul-02 <0x0A> 955.4 | Oct-02 <0x0A> 948.9', 'Rest <0x0A> 0.035 | Water placebo <0x0A> 0.049 | 4% Glucose beverage <0x0A> 0.053 | No fluid <0x0A> 0.0625', '0.9 <0x0A> 8.1 | 1.9 <0x0A> 15.0 | 2.9 <0x0A> 14.9 | 3.9 <0x0A> 13.9 | 4.9 <0x0A> 12.9 | 5.9 <0x0A> 11.0 | 6.9 <0x0A> 11.0 | 7.9 <0x0A> 10.9 | 8.9 <0x0A> 10.7 | 9.9 <0x0A> 9.9 | 10.9 <0x0A> 9.6 | 11.9 <0x0A> 4.0 | 12.9 <0x0A> 4.0 | 13.9 <0x0A> 4.0 | 14.9 <0x0A> 4.0 | 15.9 <0x0A> 4.9 | 16.9 <0x0A> 4.9 | 17.9 <0x0A> 4.9 | 18.9 <0x0A> 4.9 | 19.9 <0x0A> 4.0 | 20.9 <0x0A> 4.0']

    metric = BenetechMetric()
    res = metric(gts, pts)
        
    # metric on all batches using custom accumulation
    acc = metric.compute()
    print(f"Accuracy on all data: {acc}")

    # Reseting internal state such that metric ready for new data
    metric.reset()