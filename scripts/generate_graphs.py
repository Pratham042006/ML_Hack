"""Generate shareable PNG charts from recorded PathSnap aggregate results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'graphs'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.titleweight': 'bold', 'savefig.facecolor': 'white'})
NOTE = 'Proxy agreement with inferred labels, not verified accuracy. Student-test was inspected during development.'


def save(fig, filename):
    fig.text(.02, .015, NOTE, fontsize=8, color='#475569')
    fig.savefig(OUT / filename, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(OUT / filename)


def labels(ax, bars, suffix='%'):
    for bar in bars:
        value = bar.get_width() if hasattr(bar, 'get_width') else 0
        ax.text(value + .7, bar.get_y() + bar.get_height()/2, f'{value:.1f}{suffix}', va='center', fontsize=10)


expansion = json.loads((ROOT/'src/models/results/feature_expansion_summary.json').read_text())
two = json.loads((ROOT/'outputs/model_work/heading_comparison_v1/results.json').read_text())
two_cv = next(r['top1_including_misses'] for r in two['summary'] if r['model']=='hist_gradient_boosting' and r['variant']=='distance_heading')
counts = [2, 4, 6]
cv = [two_cv] + [next(r['top1_including_misses'] for r in expansion[k]['cv'] if r['model']=='hist_gradient_boosting') for k in ('four','six')]
dev = [0.8544, expansion['four']['student_projected']['projected']['rate'], expansion['six']['student_projected']['projected']['rate']]
fig, ax = plt.subplots(figsize=(10,5.4)); x=np.arange(3); width=.34
b1=ax.bar(x-width/2, np.array(cv)*100,width,label='Training-trip CV: classifier only',color='#2563eb')
b2=ax.bar(x+width/2,np.array(dev)*100,width,label='Student-test: projected routing',color='#0d9488')
for bars in (b1,b2): ax.bar_label(bars,fmt='%.1f%%',padding=3,fontsize=10)
ax.set_xticks(x,[f'{n} features' for n in counts]);ax.set_ylim(0,105);ax.set_ylabel('Proxy top-1 agreement, including candidate misses (%)');ax.set_title('Feature experiments: histogram gradient boosting');ax.legend(loc='upper left',fontsize=9);ax.grid(axis='y',alpha=.18);fig.tight_layout(rect=(0,.06,1,1));save(fig,'01_feature_comparison.png')

summary=json.loads((ROOT/'src/models/results/four_feature_student_summary.json').read_text())
names=['logistic_regression','gaussian_naive_bayes','decision_tree','random_forest','hist_gradient_boosting']
short=['Logistic regression','Gaussian naive Bayes','Decision tree','Random forest','Histogram boosting']
values=[summary['models'][n]['top1_including_misses']*100 for n in names]
fig,ax=plt.subplots(figsize=(10,5.4)); y=np.arange(5); bars=ax.barh(y,values,color=['#94a3b8']*4+['#2563eb']);ax.set_yticks(y,short);ax.invert_yaxis();ax.set_xlim(0,100);labels(ax,bars);ax.set_xlabel('Classifier-only proxy top-1 agreement (%)');ax.set_title('Five four-feature models: student-test comparison');ax.grid(axis='x',alpha=.18);fig.text(.02,.055,'Includes 16 candidate misses as incorrect; denominator: 9,118 confident inferred labels.',fontsize=9,color='#475569');fig.tight_layout(rect=(0,.09,1,1));save(fig,'02_five_model_comparison.png')

pairs=[('Distance + classifier',4972),('Distance + midpoint',4584),('Distance + projected',7797),('Four features + classifier',7697),('Four features + midpoint',6724),('Four features + projected',7846)]
fig,ax=plt.subplots(figsize=(10,5.8));y=np.arange(len(pairs)); values=[100*n/9118 for _,n in pairs];bars=ax.barh(y,values,color=['#94a3b8']*3+['#60a5fa','#60a5fa','#0d9488']);ax.set_yticks(y,[p[0] for p in pairs]);ax.invert_yaxis();ax.set_xlim(0,100);labels(ax,bars);ax.set_xlabel('Student-test proxy top-1 agreement (%)');ax.set_title('Scorer and routing comparison');ax.grid(axis='x',alpha=.18);fig.text(.02,.055,'Same 9,118 inferred-label denominator; projected routing produced 126 resets.',fontsize=9,color='#475569');fig.tight_layout(rect=(0,.09,1,1));save(fig,'03_routing_comparison.png')

fig,ax=plt.subplots(figsize=(10,4.8));vals=[9118,8529,1513]; names2=['Confident inferred labels','Ambiguous','Unavailable'];colors=['#0d9488','#f59e0b','#94a3b8'];left=0
for v,n,c in zip(vals,names2,colors):
 ax.barh([0],[v],left=left,color=c,height=.45,label=f'{n}: {v:,} ({v/19160:.1%})');left+=v
ax.set_xlim(0,19160);ax.set_yticks([]);ax.set_xlabel('Student-test GPS observations');ax.set_title('How much of student-test can be scored?');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=1,frameon=False,fontsize=10);fig.text(.02,.11,'16 of the 9,118 confident inferred labels were absent from the candidate shortlist.',fontsize=9,color='#475569');fig.tight_layout(rect=(0,.19,1,1));save(fig,'04_label_coverage.png')
