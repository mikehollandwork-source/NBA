"""Ranking backtest: does blending a lean's model probability with the player's
past hit rates at that line (last season, last 10 games) rank leans better than
the model probability alone?   python -m nba.analysis.rank_backtest

Result (2019-25, leans of board-eligible players): ranking by model probability
alone beat the blend in 7/7 seasons (2025 top-5/night: 78.1% vs 74.1%; random
order 63.6%). A logistic fit of all signals tied the model. Star first-game
unders were overstated. Kalshi agreement is untested (no price history).
Details: reports/phase2/ranking_backtest.md
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from nba.analysis import lean as L
from nba.analysis.common import load_games
from nba.config import PROC
OUT=str(PROC)+'/rank_backtest_'
P=L.baselines(pd.read_parquet(PROC/'projections_all_games_oos.parquet'))
R=pd.concat([L.pick(L.probabilities(P[P.season==y],fit=P,exclude_season=y)) for y in sorted(P.season.unique())],ignore_index=True)
R=R.dropna(subset=['lean_prob']).reset_index(drop=True)
S=L.STATS
act=np.array([R[s].to_numpy() for s in S]).T[np.arange(len(R)),[S.index(s) for s in R.lean_stat]]
R['act']=act; R['lean_hit']=np.where(R.lean_side=='over',act>R.lean_ref,act<R.lean_ref); R['push']=act==R.lean_ref
R=pd.concat([L.apply_pick_calibration(R[R.season==y],L.fit_pick_calibration(R[R.season!=y])) for y in sorted(R.season.unique())],ignore_index=True)
G=load_games(qualified_only=False).sort_values(['pid','game_date'])
G=G[G['min']>0]
# last-season facts
ls=G.groupby(['season','pid']).agg(n=('pra','size'),pra_m=('pra','mean'),pra_sd=('pra','std'),pts_m=('pts','mean'),team_ls=('team',lambda x:x.value_counts().index[0])).reset_index()
ls['season']+=1; ls['cv']=np.where(ls.n>=20,ls.pra_sd/ls.pra_m,np.nan)
R=R.merge(ls[['season','pid','cv','pts_m','team_ls']],on=['season','pid'],how='left')
# history hit rates at the lean's line, prior games only
R['hs']=np.nan; R['h10']=np.nan
Gd={p:d for p,d in G.groupby('pid')}
R['game_date']=pd.to_datetime(R.game_date)
for p,idx in R.groupby('pid').groups.items():
    if p not in Gd: continue
    g=Gd[p]; gd=pd.to_datetime(g.game_date).to_numpy(); gs=g.season.to_numpy()
    rr=R.loc[idx]
    vals={s:g[s].to_numpy() for s in S}
    hs=[];h10=[]
    for _,q in rr.iterrows():
        v=vals[q.lean_stat]; u=q.lean_side=='under'
        f=(v<q.lean_ref) if u else (v>q.lean_ref)
        m=gs==q.season-1; hs.append(f[m].mean() if m.sum()>=10 else np.nan)
        prior=np.where(gd<np.datetime64(q.game_date))[0][-10:]
        h10.append(f[prior].mean() if len(prior)>=5 else np.nan)
    R.loc[idx,'hs']=hs; R.loc[idx,'h10']=h10
R.to_parquet(OUT+'R.parquet',index=False)
print('rows',len(R))

# ---------------- evaluation
D=str(PROC)+'/rank_backtest_'
R=pd.read_parquet(D+'R.parquet')
R=R[(R.cv<=0.45)&(R.proj_min>=10)].copy()            # board-eligible, same as live
R['date']=pd.to_datetime(R.game_date).dt.date
R['new_team']=(R.team!=R.team_ls)&R.team_ls.notna()
R['star_g1_under']=(R.pts_m>=22)&(R.gp<1)&(R.lean_side=='under')
m=R.lean_prob_cal
R['s_model']=m
R['s_method']=pd.concat([m,R.hs,R.h10],axis=1).mean(1) - 0.03*(R.cv>0.40) - 0.03*R.new_team - 0.05*R.star_g1_under
R['s_hist']=pd.concat([R.hs,R.h10],axis=1).mean(1)
X=lambda d: np.column_stack([np.log(d.lean_prob_cal/(1-d.lean_prob_cal)), d.hs.fillna(.5)-.5, d.h10.fillna(.5)-.5, d.cv>0.40, d.new_team, d.star_g1_under])
R['s_learned']=np.nan
for y in sorted(R.season.unique()):
    tr=R[R.season!=y]; lr=LogisticRegression().fit(X(tr),tr.lean_hit)
    R.loc[R.season==y,'s_learned']=lr.predict_proba(X(R[R.season==y]))[:,1]
    if y==2025: print('learned weights (fit 2019-24):',dict(zip(['model','last_szn','last10','swing>.40','new_team','star_g1_under'],lr.coef_[0].round(2))))
rng=np.random.default_rng(7); R['s_random']=rng.random(len(R))
def topk(d,col,k):
    t=d.sort_values(col,ascending=False).groupby('date').head(k)
    n=len(t); h=t.lean_hit.mean(); return n,h,(h*0.8333-(1-h))*100
rows=[]
for name in ['s_model','s_method','s_learned','s_hist','s_random']:
    for k in [3,5,10]:
        for lab,d in [('2025',R[R.season==2025]),('2019-24',R[R.season<2025])]:
            n,h,roi=topk(d,name,k); rows.append((name[2:],k,lab,n,round(h*100,1),round(roi,1)))
T=pd.DataFrame(rows,columns=['rank_by','top_k_per_night','seasons','leans','hit%','roi%@-120'])
print(T.pivot_table(index=['rank_by','top_k_per_night'],columns='seasons',values=['hit%','roi%@-120']).to_string())
# all eligible leans, for reference
for lab,d in [('2025',R[R.season==2025])]:
    print('\nall eligible leans 2025:',len(d),'hit',round(d.lean_hit.mean()*100,1))
    print('by method score bucket 2025:'); b=pd.cut(d.s_method,[0,.5,.55,.6,.65,.7,1])
    print(d.groupby(b,observed=True).agg(n=('lean_hit','size'),hit=('lean_hit','mean')).round(3).to_string())
    print('by model prob bucket 2025:'); b=pd.cut(d.s_model,[0,.55,.6,.65,.7,1])
    print(d.groupby(b,observed=True).agg(n=('lean_hit','size'),hit=('lean_hit','mean')).round(3).to_string())
    # does history add anything at a fixed model prob? strong model leans split by history agreement
    s=d[d.s_model>=0.65]; ag=s.s_hist>=0.5
    print('model>=65%: history agrees',ag.sum(),round(s[ag].lean_hit.mean()*100,1),'| history disagrees',(~ag).sum(),round(s[~ag].lean_hit.mean()*100,1))
# per-season top5 model vs method
print('\nper season top5/night hit%:')
print(pd.DataFrame({y:{c:round(topk(R[R.season==y],c,5)[1]*100,1) for c in ['s_model','s_method','s_learned','s_random']} for y in sorted(R.season.unique())}).to_string())
