import numpy as np


def mcc_threshold(y,p):
    from sklearn.metrics import matthews_corrcoef
    if set(np.asarray(y).tolist()) != {0,1}: raise ValueError('Both C classes required')
    return float(max(np.linspace(.01,.99,99),key=lambda t:(matthews_corrcoef(y,p>=t),-abs(t-.5),-t)))


def evaluate(y,p,t,parents,bases):
    from sklearn.metrics import roc_auc_score,average_precision_score,matthews_corrcoef
    y,p,parents,bases = map(np.asarray,(y,p,parents,bases));p=p.astype(float)
    if set(y.tolist()) != {0,1}: raise ValueError('Both evaluation classes required')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)): raise ValueError('Invalid probability')
    local = []
    for parent in np.unique(parents):
        at = parents==parent; wins=0.; pairs=0
        for base in np.unique(bases[at]):
            take=at&(bases==base);delta=p[take&(y==1)][:,None]-p[take&(y==0)][None,:]
            wins+=np.sum(delta>0)+.5*np.sum(delta==0);pairs+=delta.size
        if pairs:local.append(wins/pairs)
    safe=np.clip(p,1e-7,1-1e-7)
    return dict(AUROC=float(roc_auc_score(y,p)),AP=float(average_precision_score(y,p)),
        MCC=float(matthews_corrcoef(y,p>=t)),local_AUROC=float(np.mean(local)) if local else None,
        Brier=float(np.mean((p-y)**2)),logloss=float(np.mean(-y*np.log(safe)-(1-y)*np.log1p(-safe))))
