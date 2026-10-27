"""Exact finite-distribution check of the notes' decomposition (natural logs).
No estimated entropy, learned model, Monte Carlo or medical claim.
"""
import itertools,json,math
from collections import defaultdict
from pathlib import Path
OUT=Path(__file__).resolve().parent/'results'

def entropy(rows,fields):
 masses=defaultdict(float)
 for r,p in rows:masses[tuple(r[k] for k in fields)]+=p
 return -sum(p*math.log(p) for p in masses.values() if p)
def cond(rows,a,b):return entropy(rows,a+b)-entropy(rows,b)
def main():
 rows=[]
 for a,b,c,e in itertools.product([0,1],repeat=4):
  p=.5*(.3 if b else .7)*(.2 if c else .8)*(.1 if e else .9)
  rows.append((dict(A=a,B=b,C=c,Y=a^b^c^e,Z=a),p))
 assert abs(sum(p for _,p in rows)-1)<1e-12
 intrinsic=cond(rows,['Y'],['A','B','C'])
 missing=cond(rows,['Y'],['A','B'])-intrinsic
 encoding=cond(rows,['Y'],['Z'])-cond(rows,['Y'],['A','B'])
 # Frozen, misspecified predictor overtrusts Z: p(Y=Z|Z)=0.9.
 cross_entropy=-sum(p*math.log(.9 if r['Y']==r['Z'] else .1) for r,p in rows)
 extraction=cross_entropy-cond(rows,['Y'],['Z'])
 terms=dict(intrinsic=intrinsic,missing_context=missing,lost_encoding=encoding,not_extracted=extraction)
 assert all(x>=-1e-12 for x in terms.values())
 residual=cross_entropy-sum(terms.values());assert abs(residual)<1e-12
 # Uniform no-image output is calibrated here. The same fixed model has a
 # NEGATIVE image gain despite strictly positive Shannon I(Y;I).
 no_image_ce=math.log(2)
 image_gain=no_image_ce-cross_entropy
 mi=entropy(rows,['Y'])-cond(rows,['Y'],['A','B'])
 assert image_gain<0 and mi>0
 result=dict(distribution='A~Bern(.5), B~Bern(.3), C~Bern(.2), E~Bern(.1), independently; I=(A,B), Z=A, Y=A xor B xor C xor E.',units='nats',terms=terms,cross_entropy=cross_entropy,decomposition_residual=residual,true_image_mutual_information=mi,fixed_predictor_image_ablation_gain=image_gain,conclusion='Ablation gain can be negative while true image information is positive. It measures this predictor and intervention, not missing evidence or an entropy component. V-usable information also requires a defined predictor family and appropriate fitting/evaluation; it is not interchangeable with Shannon terms.')
 (OUT/'theory_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
