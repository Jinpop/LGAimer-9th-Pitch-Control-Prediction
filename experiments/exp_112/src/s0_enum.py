import json,os,re
PK='<SCRATCH>/cw/pkg'
FILES={'blend.json':'blend.json','mate/model/blend.json':'mate/model/blend.json',
       'mate/model_q/meta.json':'mate/model_q/meta.json','model/meta.json':'model/meta.json'}
PY_FILES=['script.py','aa_infer.py','inseason.py','form_offset.py','form_decomp.py','splits.py',
          'mate/script.py','mate/q_infer.py','mate/v6_infer.py','mate/nn_infer.py','mate/deep_infer.py',
          'mate/inseason.py','mate/pipeline.py','mate/infer_ebm6.py','mate/ebm_core.py','mate/splits.py']
src={}
for f in PY_FILES:
    p=os.path.join(PK,f)
    if os.path.exists(p): src[f]=open(p,encoding='utf-8').read()
def leaves(o,p=''):
    if isinstance(o,dict):
        for k,v in o.items(): yield from leaves(v,p+'/'+str(k))
    elif isinstance(o,list):
        for i,v in enumerate(o): yield from leaves(v,p+f'/{i}')
    else: yield p,o
rep={}
for name,rel in FILES.items():
    d=json.load(open(os.path.join(PK,rel),encoding='utf-8'))
    nums=[(p,v) for p,v in leaves(d) if isinstance(v,(int,float)) and not isinstance(v,bool)]
    keys={}
    for p,v in nums:
        k=p.rsplit('/',1)[-1]
        keys.setdefault(k,0); keys[k]+=1
    hits={}
    for k in keys:
        pat=re.compile(r'["\']'+re.escape(k)+r'["\']')
        f=[fn for fn,s in src.items() if pat.search(s)]
        hits[k]=f
    rep[name]={'n_numeric_leaves':len(nums),
               'distinct_leaf_names':len(keys),
               'referenced_by_a_py_file':{k:v for k,v in sorted(hits.items()) if v},
               'never_referenced_count':sum(1 for k,v in hits.items() if not v)}
json.dump(rep,open('out/s0_enum.json','w'),ensure_ascii=False,indent=1)
for n,r in rep.items():
    print('===',n,'numeric leaves',r['n_numeric_leaves'],'distinct names',r['distinct_leaf_names'],
          'referenced',len(r['referenced_by_a_py_file']),'never',r['never_referenced_count'])
    for k,v in r['referenced_by_a_py_file'].items(): print('   ',k,'->',v)
