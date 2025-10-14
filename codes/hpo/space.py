import numpy as np

ENC_HIDDEN_CHOICES = np.array([64, 128, 256, 512])
PRED_HIDDEN_CHOICES = np.array([64, 128, 256, 512])
HEADS_CHOICES = np.array([2, 3, 4, 5, 6, 7, 8])

def nearest(arr, v): return int(arr[(np.abs(arr - v)).argmin()])

def bounds(model_name: str):
    if model_name == "gat":
        lb = [64, 2, 0.0, 64, 0.0, -5.0]; ub = [512, 8, 0.6, 512, 0.6, -2.0]
    else:
        lb = [64, 0.0, 64, 0.0, -5.0];     ub = [512, 0.6, 512, 0.6, -2.0]
    return np.array(lb, float), np.array(ub, float)

def decode_vector(x, model_name: str):
    x = np.asarray(x, float)
    i = 0
    if model_name == "gat":
        enc_hidden  = nearest(ENC_HIDDEN_CHOICES,  x[i]); i += 1
        heads       = nearest(HEADS_CHOICES,       x[i]); i += 1
        enc_drop    = float(np.clip(x[i], 0.0, 0.6));     i += 1
        pred_hidden = nearest(PRED_HIDDEN_CHOICES, x[i]); i += 1
        pred_drop   = float(np.clip(x[i], 0.0, 0.6));     i += 1
        log_lr      = float(np.clip(x[i], -5.0, -2.0));   i += 1
        return dict(enc_hidden=enc_hidden, heads=heads, enc_drop=enc_drop,
                    pred_hidden=pred_hidden, pred_drop=pred_drop, lr=10.0**log_lr)
    else:
        enc_hidden  = nearest(ENC_HIDDEN_CHOICES,  x[i]); i += 1
        enc_drop    = float(np.clip(x[i], 0.0, 0.6));     i += 1
        pred_hidden = nearest(PRED_HIDDEN_CHOICES, x[i]); i += 1
        pred_drop   = float(np.clip(x[i], 0.0, 0.6));     i += 1
        log_lr      = float(np.clip(x[i], -5.0, -2.0));   i += 1
        return dict(enc_hidden=enc_hidden, enc_drop=enc_drop,
                    pred_hidden=pred_hidden, pred_drop=pred_drop, lr=10.0**log_lr)
