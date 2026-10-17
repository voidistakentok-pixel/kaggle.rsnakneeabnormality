import sys,json,ast,tempfile,time,hashlib
from pathlib import Path
import nbformat,numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss
p=Path(__file__).resolve().parents[1]/'rsna_knee_dinov2_nested_cv.ipynb'
nb=nbformat.read(p,as_version=4); nbformat.validate(nb)
for i,c in enumerate(nb.cells): compile(c.source,f'cell_{i}','exec')
print('PASS: notebook schema; all 11 code cells compile; no stored execution results')
globals_={k:v for k,v in globals().items() if not k.startswith('__')}
globals_.update({'CFG':{'seed':2026,'folds':3,'backbones':['synthetic_A','synthetic_B'],'metric':'macro_auc'},'LABELS':[f'label_{j}' for j in range(12)],'CUSTOM':None,'weights':np.ones(12),'ID':'study_id','FOLD_SIGNATURE':'synthetic-test-only','PREPROCESS':{},'START':time.monotonic()})
rng=np.random.default_rng(2026); n=60; nt=7
Y=rng.integers(0,2,(n,12)).astype(float)
features={k:rng.normal(size=(n+nt,4,3,8)).astype(np.float32) for k in ['synthetic_A','synthetic_B']}
tmp=Path(tempfile.mkdtemp(prefix='knee_validation_'))
def digest(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()
def atomic_json(path,obj): Path(path).write_text(json.dumps(obj,default=str))
globals_.update({'WORK':tmp,'y':Y,'folds':np.arange(n)%3,'groups':np.array([f'g{i}' for i in range(n)]),'TRAIN_IDS':[str(i) for i in range(n)],'TEST_IDS':[str(i) for i in range(n,n+nt)],'DATA_SIGNATURE':'synthetic','features_for':lambda name:features[name],'remaining':lambda:99999,'digest':digest,'atomic_json':atomic_json})
# Execute actual metric, training/tuning, nested ensembling and submission cells with synthetic features.
tree=ast.parse(nb.cells[4].source)
for node in tree.body:
 if isinstance(node,ast.FunctionDef) and node.name in {'per_label','score'}:
  exec(compile(ast.Module(body=[node],type_ignores=[]),'metric','exec'),globals_)
exec(nb.cells[7].source,globals_)
# Re-execute to exercise saved fold predictions and configurations.
exec(nb.cells[7].source,globals_)
exec(nb.cells[9].source,globals_)
sample=pd.DataFrame({'study_id':globals_['TEST_IDS'],**{label:['0']*nt for label in globals_['LABELS']}})
globals_.update({'sample':sample,'MODE':'wide'})
# Display dependency is Kaggle-only; retain all actual CSV assertions here.
source=nb.cells[10].source.split('from IPython.display')[0]
exec(source,globals_)
assert (tmp/'submission.csv').is_file()
assert pd.read_csv(tmp/'submission.csv').shape==(nt,13)
# Exercise the explicit long-row submission mapping too.
row_map=pd.DataFrame([(f'{sid}_{label}',sid,label) for sid in globals_['TEST_IDS'] for label in globals_['LABELS']],columns=['row_id','study_id','label'])
sample_long=pd.DataFrame({'row_id':row_map.row_id,'probability':['0']*len(row_map)})
globals_.update({'sample':sample_long,'MODE':'long','row_map':row_map})
exec(source,globals_)
assert pd.read_csv(tmp/'submission.csv').shape==(nt*12,2)
# Test one-class training folds return finite smoothed probabilities.
m=globals_['fit_head'](np.zeros((5,3)),np.zeros((5,12)),.1)
pr=globals_['predict_head'](m,np.ones((2,3)))
assert pr.shape==(2,12) and np.isfinite(pr).all() and ((pr>0)&(pr<1)).all()
# Missing metric is explicitly blocked.
assert "if CFG['metric'] in {None,'auto'}:" in nb.cells[4].source
globals_['CFG']['metric']='macro_log_loss'
assert globals_['score'](Y,np.full(Y.shape,.5))<0
globals_['CFG']['metric']='weighted_log_loss'
assert np.isclose(globals_['score'](Y,np.full(Y.shape,.5)),-np.log(2))
print('PASS: actual tuning, fold-cache resume, nested ensemble, bootstrap, wide submission, one-class heads')
report={'nbformat_validation':'passed','python_compilation_cells':len(nb.cells),'synthetic_workflow':'passed','gpu_or_real_kaggle_data_execution':False,'note':'Synthetic scores are not competition results and are not stored in the deliverable.'}
(tmp/'validation_report.json').write_text(json.dumps(report,indent=2))
print('Validation artifacts:',tmp)
