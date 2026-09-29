"""Numerical cross-checks with independently sampled transverse polarizations."""
import numpy as np
import sympy as sp

from data_gen.data_gen_ym.kinematics import generate_kinematics
from .algebra import LABELS

def make_points(seed, count):
    if count < 1:
        raise ValueError('samples per mode must be positive')
    points=[]; errors=[]
    metric = np.array([1., -1., -1., -1.])
    for mode in ('coulomb', 'covariant'):
        for j in range(count):
            point_seed=seed+j+(1000 if mode=='covariant' else 0)
            mom,_=generate_kinematics(5,seed=point_seed)
            rng=np.random.default_rng(point_seed+100000)
            pol=np.zeros_like(mom)
            # Independent random linear transverse polarizations avoid the
            # fixed reference-axis correlations in the inference evaluator.
            for i,p in enumerate(mom):
                v=rng.normal(size=3)
                v-=p[1:]*np.dot(v,p[1:])/np.dot(p[1:],p[1:])
                pol[i,1:]=v/np.linalg.norm(v)
                if mode=='covariant': pol[i]+=rng.uniform(-.7,.7)*p
            errors.append(max(np.max(np.abs(np.sum(mom,axis=0))),
                              np.max(np.abs(np.sum(mom*mom*metric,axis=1))),
                              np.max(np.abs(np.sum(pol*mom*metric,axis=1)))))
            if errors[-1] >= 1e-10:
                raise RuntimeError('Generated point violates on-shell constraints')
            points.append((mom,pol))
    return points, max(errors)

def evaluate(expr, points):
    symbols=sorted(expr.free_symbols,key=str)
    function=sp.lambdify(symbols,expr,modules='numpy',cse=True)
    values=[]
    for mom,pol in points:
        vals=[]
        for s in symbols:
            (a,i),(b,j)=LABELS[s]
            x=(mom if a=='p' else pol)[i-1]
            y=(mom if b=='p' else pol)[j-1]
            vals.append(x[0]*y[0]-np.dot(x[1:],y[1:]))
        value=float(function(*vals))
        if not np.isfinite(value): raise ValueError('nonfinite numerical evaluation')
        values.append(value)
    return np.array(values)

def compare(a,b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not a.size or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('Numerical comparisons need equal, nonempty, finite arrays')
    diffs=np.abs(a-b)
    scales=np.maximum(1,np.maximum(np.abs(a),np.abs(b)))
    return {'passes':bool(np.all(diffs<=1e-9+1e-8*np.maximum(np.abs(a),np.abs(b)))),
            'max_absolute_error':float(np.max(diffs)),
            'max_scaled_error':float(np.max(diffs/scales))}
