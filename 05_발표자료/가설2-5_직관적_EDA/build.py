"""Fresh CSV aggregation; pandas/numpy/matplotlib only, plus Python standard library.
No fitted model, scipy, sklearn, image generation, or precomputed outcomes.
"""
from pathlib import Path
import sys
import json
import hashlib
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
FIG=OUT/'figures'
FIG.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'대시보드'))
# Existing cleaning/identity rules use only pandas/numpy and the standard library.
from housing.data import prepare_data, READ_COLUMNS

BLUE='#2878b5'; ORANGE='#e48b32'; GREEN='#329a79'; TEXT='#172a3a'; MUTED='#647587'
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'figure.facecolor':'white','axes.facecolor':'white','text.color':TEXT,'axes.labelcolor':TEXT,'xtick.color':MUTED,'ytick.color':MUTED,'font.size':11})
KINDS=['지하철','학교','공원','대규모점포']
FEATURES=['activity','households','age','parking']+KINDS

def export(t,name):t.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
def md(t):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:,.1f}' if np.isfinite(v) else '결측'
        return str(v)
    return '\n'.join(['| '+' | '.join(t.columns)+' |','| '+' | '.join(['---']*len(t.columns))+' |']+['| '+' | '.join(fmt(v) for v in r)+' |' for r in t.itertuples(index=False,name=None)])

def canvas(title,subtitle):
    f=plt.figure(figsize=(16,9),dpi=120)
    f.text(.06,.93,title,fontsize=26,fontweight='bold')
    f.text(.06,.875,subtitle,fontsize=12,color=MUTED)
    return f

def style(ax):
    ax.grid(axis='y',alpha=.18,zorder=0)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['bottom','left']].set_color('#dce2e7')

def save(f,stem,msg,note):
    f.text(.06,.12,msg,fontsize=17,fontweight='bold',bbox=dict(boxstyle='round,pad=.65',facecolor='#eef5fa',edgecolor='none'))
    f.text(.06,.056,note,fontsize=10,color=MUTED)
    f.text(.06,.023,'서울 아파트 실거래 CSV 직접 재집계 | 전용 60㎡ 이상 85㎡ 미만 | 기준시점 2026Q2 | 인과관계가 아닌 탐색 비교',fontsize=9,color=MUTED)
    f.savefig(FIG/(stem+'.png'),dpi=120)
    f.savefig(FIG/(stem+'.svg'))
    plt.close(f)

def load():
    source=ROOT/'거래_단지_DATA/아파트_거래_DATA_대시보드_실데이터.csv'
    if '--render-only' in sys.argv:
        metadata=json.loads((OUT/'분석출처.json').read_text(encoding='utf-8'))
        with source.open('rb') as h: assert hashlib.file_digest(h,'sha256').hexdigest()==metadata['sha256']
        d=pd.read_csv(OUT/'원천재집계_하락관측표본.csv')
        d=d[~np.isclose(d.loss,0)].copy()
        export(d,'원천재집계_하락관측표본')
        metadata['decline_n']=len(d)
        (OUT/'분석출처.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        return d[d.loss.ge(5)].copy(),metadata
    cols=pd.read_csv(source,nrows=0).columns
    raw=pd.read_csv(source,usecols=[c for c in READ_COLUMNS if c in cols],dtype=str,keep_default_na=False)
    print('Source rows:',len(raw),flush=True)
    data=prepare_data(raw,source.name)
    del raw
    q=data.quarterly
    q=q[q.area_group.eq('60㎡ 이상 85㎡ 미만') & q.quarter.le(pd.Period('2026Q2'))]
    rows=[]
    for cid,g in q.groupby('complex_id'):
        g=g.sort_values('quarter');v=g[g['count'].ge(3)&g.complete]
        if len(v)<4 or v.quarter.iloc[-1]!=pd.Period('2026Q2'):continue
        p=v.price.to_numpy(); h=np.maximum.accumulate(p); dd=(p/h-1)*100;j=int(np.argmin(dd))
        if dd[j]>=0 or np.isclose(dd[j],0):continue
        rec=(p[-1]-p[j])/(h[j]-p[j])*100
        rows.append([cid,-dd[j],rec,len(g),len(v),int(g['count'].sum()),float(g['count'].sum()/len(g))])
    d=pd.DataFrame(rows,columns=['complex_id','loss','recovery','observed_quarters','qualified_quarters','trades','activity']).merge(data.complexes,on='complex_id',validate='one_to_one')
    facilities=data.facilities[data.facilities.status.eq('ok')].pivot(index='complex_id',columns='kind',values='distance')
    d=d.merge(facilities,left_on='complex_id',right_index=True,how='left',validate='one_to_one')
    d['age']=2026-d.built_year
    d['parking']=(d.parking_ground+d.parking_underground)/d.households
    assert d.complex_id.is_unique
    export(d,'원천재집계_하락관측표본')
    with source.open('rb') as h:sha=hashlib.file_digest(h,'sha256').hexdigest()
    metadata={'source':str(source),'sha256':sha,'diagnostics':data.diagnostics,'start':str(data.start),'end':str(data.end),'decline_n':len(d),'analysis_n':int(d.loss.ge(5).sum()),'libraries':['pandas','numpy','matplotlib','Python standard library']}
    (OUT/'분석출처.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return d[d.loss.ge(5)].copy(),metadata

def balanced_groups(d,feature,labels):
    result=[]
    for gu,g in d.dropna(subset=[feature]).groupby('district'):
        g=g.sort_values([feature,'complex_id']); n=len(g)//len(labels)
        if n<1:continue
        sets=[g.iloc[:n],g.iloc[(len(g)-n)//2:(len(g)-n)//2+n],g.iloc[-n:]] if len(labels)==3 else [g.iloc[:n],g.iloc[-n:]]
        for label,part in zip(labels,sets):
            part=part.copy();part['group']=label;result.append(part)
    r=pd.concat(result)
    assert r.complex_id.is_unique
    c=r.groupby(['district','group']).size().unstack()
    assert c.nunique(axis=1).eq(1).all()
    return r

def summary(g,feature,labels):
    return pd.DataFrame([[label,len(g[g.group.eq(label)]),g.loc[g.group.eq(label),feature].median(),g.loc[g.group.eq(label),'loss'].median(),g.loc[g.group.eq(label),'recovery'].median(),g.loc[g.group.eq(label),'qualified_quarters'].median()] for label in labels],columns=['집단','단지수','특성중앙값','낙폭중앙값(%)','하락회복률중앙값(%)','유효분기수중앙값'])

def group_plot(d,feature,labels,title,stem,feature_unit):
    g=balanced_groups(d,feature,labels);t=summary(g,feature,labels)
    f=canvas(title,f'각 자치구에서 {labels[0]}·{labels[1]}·{labels[2]} 집단의 수를 같게 비교 | 집단별 {int(t.단지수.iloc[0])}개')
    axs=f.subplots(1,2,gridspec_kw=dict(left=.08,right=.94,top=.77,bottom=.29,wspace=.25))
    for ax,col,head in [(axs[0],'낙폭중앙값(%)','얼마나 떨어졌나?  ↓ 낮을수록 방어력이 좋음'),(axs[1],'하락회복률중앙값(%)','하락분을 얼마나 되찾았나?  ↑ 높을수록 좋음')]:
        ax.bar(range(3),t[col],color=[BLUE,ORANGE,GREEN],width=.54,zorder=3)
        top=max(t[col].max()*1.27,35 if col.startswith('낙폭') else 130)
        ax.set_ylim(0,top)
        for i,v in enumerate(t[col]):ax.text(i,v+top*.025,f'{v:.1f}%',ha='center',fontweight='bold',fontsize=23)
        shown=-t.특성중앙값 if feature=='newness' else t.특성중앙값
        ax.set_xticks(range(3),[f'{lab}\n{feature_unit} {v:.1f}' for lab,v in zip(labels,shown)])
        ax.set_title(head,loc='left',fontsize=14,pad=15);style(ax)
        if col.startswith('하락'):ax.axhline(100,color=MUTED,ls='--',alpha=.7);ax.text(2.58,100,'100%\n고점 회복',ha='left',va='center',fontsize=9,color=MUTED)
        ax.set_xlim(-.6,2.9)
    a,b=t.iloc[0],t.iloc[-1]
    msg=f"{labels[-1]} 집단: 낙폭 {b.iloc[3]:.1f}%, 회복률 {b.iloc[4]:.1f}% — 두 지표를 함께 확인해야 한다."
    note='막대는 집단 중앙값. 구별 순위로 나눈 상대적 집단이며 다른 특성·고점/저점 시기까지 같게 맞춘 비교는 아님.'
    if feature=='activity': note=f'거래 적은/많은 집단의 유효 관측분기 중앙값: {a.iloc[5]:.0f}개 / {b.iloc[5]:.0f}개. 관측 기회 차이도 낙폭 비교에 영향을 줄 수 있음.'
    save(f,stem,msg,note)
    export(t,stem+'_숫자');export(g[['complex_id','district','name','group',feature,'loss','recovery','qualified_quarters']],stem+'_집단구성')
    return t,g

def location(d):
    rows=[];members=[]
    for kind in KINDS:
        g=balanced_groups(d,kind,['가까운 편','먼 편'])
        for lab in ['가까운 편','먼 편']:
            s=g[g.group.eq(lab)]
            rows.append([kind,lab,len(s),s[kind].median(),s.loss.median(),s.recovery.median()])
        members.append(g[['complex_id','district','group',kind]].rename(columns={kind:'distance'}).assign(kind=kind))
    t=pd.DataFrame(rows,columns=['시설','집단','단지수','거리중앙값(m)','낙폭중앙값(%)','하락회복률중앙값(%)'])
    f=canvas('가설 4  같은 구에서 시설이 가까운 단지가 더 잘 버텼을까?','시설별로 같은 자치구 안의 가까운 절반·먼 절반을 같은 수로 비교 | 두 집단의 자치구 구성 동일')
    axs=f.subplots(1,2,gridspec_kw=dict(left=.08,right=.94,top=.75,bottom=.29,wspace=.24))
    for ax,col,title in [(axs[0],'낙폭중앙값(%)','낙폭 중앙값 · 낮을수록 좋음'),(axs[1],'하락회복률중앙값(%)','회복률 중앙값 · 높을수록 좋음')]:
        for delta,lab,color in [(-.19,'가까운 편',BLUE),(.19,'먼 편',ORANGE)]:
            vals=t[t.집단.eq(lab)][col].to_numpy();x=np.arange(4)+delta
            ax.bar(x,vals,width=.35,color=color,label=lab,zorder=3)
            for xx,v in zip(x,vals):ax.text(xx,v+max(t[col])*.025,f'{v:.1f}',ha='center',fontsize=11,fontweight='bold')
        ax.set_ylim(0,max(t[col])*1.25);ax.set_xticks(range(4),[f'{k}\n각 {n}개' for k,n in zip(KINDS,t[t.집단.eq('가까운 편')].단지수)]);ax.set_title(title,loc='left',fontsize=14,pad=15);style(ax)
        if col.startswith('하락'):ax.axhline(100,color=MUTED,ls='--',alpha=.6)
    axs[0].legend(frameon=False,loc='upper left',ncol=2,bbox_to_anchor=(0,1.17))
    save(f,'04_가설4_시설거리','가까운 집단의 회복률은 높았지만, 낙폭이 더 작지는 않았다.','가까움/멂은 구 안에서의 상대적 구분. 시설명·단지좌표가 있고 거리 0~1,000m인 값만 사용. 단위는 %.')
    export(t,'04_가설4_숫자');export(pd.concat(members),'04_가설4_집단구성');return t

def pairs(d):
    s=d.dropna(subset=FEATURES).copy();s=s[(s.activity>0)&(s.households>0)&(s.parking>=0)]
    candidates=[]
    for gu,g in s.groupby('district'):
        a=g[FEATURES].to_numpy(); ids=g.index.to_numpy()
        for i in range(len(g)):
            for j in range(i+1,len(g)):
                v,w=a[i],a[j];delta=np.abs(v-w)
                scaled=np.r_[np.abs(np.log(v[:2]/w[:2]))/np.log(1.5),delta[2:]/[5,.3,200,200,200,200]]
                if (scaled<=1+1e-12).all():candidates.append((float(np.mean(scaled**2)),str(g.iloc[i].complex_id),str(g.iloc[j].complex_id),ids[i],ids[j]))
    used=set();chosen=[]
    for score,cid1,cid2,i,j in sorted(candidates):
        if i in used or j in used:continue
        used.update([i,j]);chosen.append((score,i,j))
    rows=[];details=[]
    for k,(score,i,j) in enumerate(chosen,1):
        a,b=d.loc[i],d.loc[j]
        rows.append([k,a.district,a['name'],b['name'],score,a.loss,b.loss,abs(a.loss-b.loss),a.recovery,b.recovery,abs(a.recovery-b.recovery)])
        for lab,r in [('A',a),('B',b)]:details.append([k,lab,r.complex_id,r.district,r['name']]+[r[c] for c in FEATURES]+[r.loss,r.recovery])
    t=pd.DataFrame(rows,columns=['쌍번호','자치구','단지A','단지B','특성차이점수','A낙폭(%)','B낙폭(%)','낙폭차이(%p)','A회복률(%)','B회복률(%)','회복률차이(%p)'])
    dt=pd.DataFrame(details,columns=['쌍번호','구분','complex_id','district','name']+FEATURES+['loss','recovery'])
    assert len(t)>0 and dt.complex_id.is_unique
    f=canvas('가설 5  여러 조건이 비슷한 단지끼리도 차이가 남을까?',f'8개 특성이 모두 확인된 {len(s)}개 → 조건을 충족한 중복 없는 {len(t)}쌍 | 가격 결과를 보지 않고 특성으로만 짝 선정')
    axs=f.subplots(1,2,gridspec_kw=dict(left=.08,right=.94,top=.77,bottom=.29,wspace=.24))
    for ax,col,title,color in [(axs[0],'낙폭차이(%p)','두 단지의 낙폭 차이 (%p)',BLUE),(axs[1],'회복률차이(%p)','두 단지의 회복률 차이 (%p)',ORANGE)]:
        bins=[0,2,5,10,np.inf] if col.startswith('낙폭') else [0,50,100,300,np.inf]
        labs=['2 미만','2~5 미만','5~10 미만','10 이상'] if col.startswith('낙폭') else ['50 미만','50~100 미만','100~300 미만','300 이상']
        counts=pd.cut(t[col],bins,right=False,labels=labs).value_counts(sort=False)
        ax.bar(range(4),counts,color=color,width=.6,zorder=3)
        for j,v in enumerate(counts):ax.text(j,v+.25,f'{v}쌍',ha='center',fontsize=22,fontweight='bold')
        med=t[col].median();ax.text(.98,.95,f'차이 중앙값 {med:.1f}%p',ha='right',va='top',transform=ax.transAxes,fontsize=18,fontweight='bold')
        ax.set_ylim(0,max(counts.max()*1.45,10));ax.set_yticks(range(0,int(ax.get_ylim()[1])+1,2));ax.set_xticks(range(4),labs);ax.set_xlabel('두 단지 사이 차이 구간 (%p)');ax.set_ylabel('해당하는 단지쌍 수');ax.set_title(title,loc='left',fontsize=14,pad=15);style(ax)
    save(f,'05_가설5_비슷한조건_단지쌍',f"비슷한 조건의 {len(t)}쌍에서도 차이가 관측됐다. 다만 모든 조건이 동일한 비교는 아니다.",'조건: 거래량·세대수 비율 ≤1.5배 / 연식 차이 ≤5년 / 세대당 주차 ≤0.3대 / 시설별 거리 차이 ≤200m.')
    export(t,'05_가설5_짝별_숫자');export(dt,'05_가설5_8개특성_대조표')
    return t,dt,len(s),len(candidates)

def report(d,meta,ts,loc,pt,details,complete,candidates):
    counts=[]
    for col,bins,labs in [('낙폭차이(%p)',[0,2,5,10,np.inf],['2 미만','2~5 미만','5~10 미만','10 이상']),('회복률차이(%p)',[0,50,100,300,np.inf],['50 미만','50~100 미만','100~300 미만','300 이상'])]:
        for label,n in pd.cut(pt[col],bins,right=False,labels=labs).value_counts(sort=False).items(): counts.append([col,label,int(n)])
    count_table=pd.DataFrame(counts,columns=['차이지표','차이구간(%p)','단지쌍수']);export(count_table,'05_가설5_차이구간_숫자')
    labels=['가설 2 — 거래가 활발한 단지일수록 더 잘 방어하고 회복할까?','가설 3-1 — 세대수가 많은 단지일수록 더 잘 방어하고 회복할까?','가설 3-2 — 신축에 가까운 단지일수록 더 잘 방어하고 회복할까?']
    stems=['01_가설2_거래활발도','02_가설3-1_세대수','03_가설3-2_연식']
    units=['거래가 관측된 분기당 평균 거래수(건)','세대수(세대)','연식(년, 2026-건축연도)']
    chunks=['# 가설 2~5: 직관적 시각화와 장표용 숫자\n',f'''## 공통 분석 기준

- 지정 CSV를 새로 읽어 거래 가격부터 재집계했다. 원천 거래 {meta['diagnostics']['원본 행']:,}건, 단지 {meta['diagnostics']['고유 단지']:,}개. CSV·대시보드 파일은 변경하지 않았다.
- 분석·그래프 라이브러리는 pandas, NumPy, Matplotlib만 사용했다. 경로·해시·압축 등에는 파이썬 표준 라이브러리를 사용했다. 기존 대시보드의 동일 단지 식별·전처리 함수도 pandas·NumPy 기반이다. 회귀·머신러닝·외부 통계 라이브러리는 사용하지 않았다.
- 전용 **60㎡ 이상 85㎡ 미만**, **2026Q2** 기준. 가격은 거래금액/전용면적의 단지·분기 중앙값(만원/㎡). 거래 3건 이상인 완결 분기 4개 이상, Q2 거래 3건 이상. 연속 4분기라는 뜻은 아니다.
- 하락이 관측된 {meta['decline_n']}개 중 **낙폭 5% 이상 {len(d)}개**로 분석했다. 작은 낙폭에서 회복률 분모가 작아지는 문제를 줄이기 위한 탐색 조건이며 전체 시장의 대표 표본은 아니다.
- MDD 절대폭은 **작을수록 방어력이 좋음**. 음수 MDD -20%는 그림에서 낙폭 20%로 표현한다. 하락회복률은 (Q2가격-저점)/(MDD구간 고점-저점)×100. **100%는 그 고점 회복**, 150%는 하락분의 1.5배 회복이다. 상승률 150%나 빠른 회복을 뜻하지 않는다.
- 가설 2·3: 각 자치구에서 해당 특성으로 정렬하고 작은/중간/큰 집단에 같은 수를 배정했다. 각 집단은 floor(유효 단지수/3)개이고 나머지 최대 2개는 제외한다. 동률은 단지 ID로 순서를 정한다. 따라서 일부 경계 동률이 집단에 나뉠 수 있다. **집단별 자치구 구성은 같지만 다른 특성까지 같지는 않다.**
- 가설 4: 시설별로 각 구의 가까운 절반과 먼 절반에서 같은 수를 선택한다. 홀수일 때 가운데 1개는 제외한다. 같은 구 안에서 가까운 편이라는 뜻이지 절대적인 역세권 기준은 아니다.
- 막대는 집단 전체의 **중앙값**이다. 구별 효과를 추정한 회귀계수나 개별 단지의 순수한 특성 효과가 아니다. 구별 구성 균형을 맞췄더라도 구 내부의 교란이나 중앙값 집계의 한계는 남는다.
- 같은 면적군이지만 정확한 면적·층·동은 다를 수 있다. 고점·저점 시점, 회복에 주어진 시간도 다르다. 통계적 유의성·인과관계·미래 예측력을 검증한 자료가 아니다.
''']
    for title,stem,t,unit in zip(labels,stems,ts,units):
        a,b=t.iloc[0],t.iloc[-1]
        supported=b.iloc[3]<a.iloc[3] and b.iloc[4]>a.iloc[4]
        direction='양쪽 지표가 예상 방향으로 움직였지만, 인과관계를 뜻하지는 않는다.' if supported else '두 지표가 예상 방향으로 함께 움직이지 않아, 이 특성만으로 방어·회복 모두 좋다고 판단하기 어렵다.'
        # Age order is old -> middle -> new, so the last group is the hypothesis direction.
        chunks.append(f'''## {title}

![{title}](figures/{stem}.png)

{md(t)}

특성중앙값 단위: **{unit}**. 유효분기 수는 거래 3건 이상인 분기 수다.

**장표 핵심 숫자**

- 집단별 **{int(a.iloc[1])}개**를 비교했다.
- {a.iloc[0]}: 낙폭 **{a.iloc[3]:.1f}%**, 하락회복률 **{a.iloc[4]:.1f}%**.
- {b.iloc[0]}: 낙폭 **{b.iloc[3]:.1f}%**, 하락회복률 **{b.iloc[4]:.1f}%**.
- 두 집단 간 차이({b.iloc[0]}-{a.iloc[0]}): 낙폭 **{b.iloc[3]-a.iloc[3]:+.1f}%p**, 회복률 **{b.iloc[4]-a.iloc[4]:+.1f}%p**.

**발표 멘트:** “각 구에서 비교 집단의 수를 같게 맞춘 뒤 실제 낙폭과 회복률을 비교했습니다. {a.iloc[0]}의 낙폭과 회복률은 {a.iloc[3]:.1f}%, {a.iloc[4]:.1f}%였고, {b.iloc[0]}은 {b.iloc[3]:.1f}%, {b.iloc[4]:.1f}%였습니다. {direction}”

**해석할 때:** 거래 활발도는 전체 거래수/거래가 한 번이라도 있는 관측분기 수다. 거래 없는 분기를 포함한 달력 평균은 아니다. 거래 기회가 많으면 하락을 포착할 기회도 많고, 신축은 관측 이력이 짧을 수 있다. 세대수·연식은 관리 품질이나 시장 선호도 자체를 직접 측정하지 않는다.
''')
    chunks.append(f'''## 가설 4 — 교통·생활시설이 가까우면 더 잘 버티고 회복할까?

![가설 4](figures/04_가설4_시설거리.png)

{md(loc)}

**장표용 읽는 방법:** 파란 막대가 가까운 편, 주황 막대가 먼 편이다. 왼쪽에서는 파란 막대가 낮아야 가설 방향이고, 오른쪽에서는 파란 막대가 높아야 가설 방향이다. 점수가 아니라 실제 낙폭·회복률 중앙값이다.

**시설별 숫자와 해석**
''')
    for kind in KINDS:
        a=loc[(loc.시설==kind)&(loc.집단=='가까운 편')].iloc[0];b=loc[(loc.시설==kind)&(loc.집단=='먼 편')].iloc[0]
        direction='두 지표가 모두 가설 방향' if a.iloc[4]<b.iloc[4] and a.iloc[5]>b.iloc[5] else '두 지표 모두에서 일관된 우위는 아님'
        chunks.append(f"- **{kind}**, 집단별 {a.iloc[2]}개: 가까운 편 거리 중앙값 {a.iloc[3]:.0f}m / 낙폭 {a.iloc[4]:.1f}% / 회복률 {a.iloc[5]:.1f}%; 먼 편 {b.iloc[3]:.0f}m / {b.iloc[4]:.1f}% / {b.iloc[5]:.1f}%. **{direction}**.\n")
    chunks.append(f'''\n**발표 멘트:** “시설이 가까운 편과 먼 편을 같은 구 안에서 같은 수로 나눴습니다. 시설별 실제 낙폭과 회복률을 나란히 보면 두 지표를 따로 확인할 필요가 있습니다. 이 결과를 입지가 중요하지 않다는 뜻으로 해석하지 않았습니다.”

**한계:** 현재 매핑 규칙상 모든 시설에서 시설명이 있고 단지 좌표가 유효하며 거리 0~1,000m인 값만 사용했다. 먼 편도 이 범위 안의 상대적 구분이다. 특히 공원 결측이 많다. 거리만으로 학군·환승·상권·시설 품질·도보시간을 측정하지 못하며 시설별 표본도 다르다.

## 가설 5 — 여러 특성이 비슷한 단지쌍에서도 실제 결과는 다를까?

![가설 5](figures/05_가설5_비슷한조건_단지쌍.png)

**방법:** 회귀나 예측모형을 사용하지 않고, 같은 자치구·같은 면적군에서 아래 조건을 모두 만족하는 단지쌍을 비교했다.

| 비교 특성 | 허용한 차이 |
| --- | --- |
| 거래가 있는 분기당 거래수 | 큰 값/작은 값 ≤1.5배 |
| 세대수 | 큰 값/작은 값 ≤1.5배 |
| 연식 | 차이 ≤5년 |
| 세대당 주차대수 | 차이 ≤0.3대 |
| 지하철·학교·공원·대규모점포 거리 | 시설 각각 차이 ≤200m |

8개 특성이 모두 있는 **{complete}개**에서 조건을 충족하는 후보는 **{candidates}쌍**이었다. 특성 차이가 작은 순으로 고르고 이미 선택된 단지는 다시 사용하지 않아 최종 **{len(pt)}쌍 / {len(pt)*2}개 단지 / {pt.자치구.nunique()}개 구**를 비교했다. 거래량·세대수는 로그 비율/로그1.5, 나머지는 차이/허용차이로 정규화하고 제곱 평균이 작은 순으로 선택했다. 가격 결과는 짝 선정이나 순서에 사용하지 않았다. 이 허용치는 탐색용이며 검증된 동일 상품 기준이 아니다.

**장표 핵심 숫자**

- 두 단지 간 낙폭 차이 중앙값: **{pt['낙폭차이(%p)'].median():.1f}%p**.
- 두 단지 간 하락회복률 차이 중앙값: **{pt['회복률차이(%p)'].median():.1f}%p**.
- 낙폭 차이 범위: **{pt['낙폭차이(%p)'].min():.1f}~{pt['낙폭차이(%p)'].max():.1f}%p**.
- 회복률 차이 범위: **{pt['회복률차이(%p)'].min():.1f}~{pt['회복률차이(%p)'].max():.1f}%p**.

각 쌍에서 **두 단지 지표 차이의 절댓값**을 구한 뒤 구간으로 묶었다. 막대 높이는 해당 차이 구간에 들어가는 **단지쌍 수**다. 예를 들어 회복률 100~300%p 막대는 그만큼 회복률이 다른 쌍의 개수를 뜻한다. 구간 너비가 다르므로 연속 히스토그램의 면적이 아니라 범주별 개수를 읽는다. 아래 표 번호는 특성이 가까운 순이며 결과가 좋은 순이나 차이가 큰 순이 아니다.

{md(count_table)}

{md(pt.drop(columns='특성차이점수'))}

**발표 멘트:** “거래량, 세대수, 연식, 주차, 시설 거리가 비교적 비슷한 단지쌍을 골랐습니다. {len(pt)}쌍을 비교했을 때 낙폭 차이의 중앙값은 {pt['낙폭차이(%p)'].median():.1f}%p, 회복률 차이의 중앙값은 {pt['회복률차이(%p)'].median():.1f}%p였습니다. 비슷한 특성만으로 두 단지의 가격 경로까지 같다고 판단할 수는 없었습니다.”

**중요한 해석 범위:** 이 그림은 원래 가설 5를 ‘유사 조건에서 관측 차이가 남는가’라는 탐색 질문으로 바꿔 확인한 것이다. 다변량 통계적 통제나 인과 효과 추정이 아니다. 허용 범위 내에서도 특성이 다르고, 관측기간·고점·저점 시기·정확한 면적·층은 같게 맞추지 않았다. 결측 없는 소수 표본에 국한된다. 거래 잡음과 누락 요인도 차이에 포함되므로 차이가 남았다는 것만으로 단지 고유 효과나 회복 능력이 증명되지는 않는다.

## 서비스 시연으로 이어지는 문장

> 같은 자치구·같은 면적군에서도 단지별 하락과 회복은 달랐습니다. 거래량·규모·연식·입지로 나누어 보고 비슷한 조건의 단지끼리도 비교했지만, 몇 가지 특성만으로 실제 가격 흐름까지 같다고 볼 수는 없었습니다. 그래서 관심 단지의 가격 흐름과 MDD·하락회복률, 단지 특성을 직접 비교할 수 있는 서비스를 만들었습니다.

## 파일

- `figures/`: 발표용 PNG 5장(1920×1080), 확대용 SVG 5장.
- `*_숫자.csv`: 그림과 동일한 원수치. `05_가설5_8개특성_대조표.csv`: 짝별 실제 조건을 검토할 수 있는 표.
- 이전 버전은 자치구 대비 차이·보정상관·회귀모형을 사용했고 이번 버전은 균형 집단의 원지표 중앙값·유사 단지쌍 비교를 사용했다. 질문·표본·계산법이 달라 숫자가 달라질 수 있으므로 장표끼리 섞지 않는다.
''')
    (OUT/'가설2-5_숫자장표_해석_발표대본.md').write_text('\n'.join(chunks),encoding='utf-8')

def main():
    d,meta=load();print('Analysis sample:',len(d),flush=True)
    t2,_=group_plot(d,'activity',['거래 적은 편','중간','거래 많은 편'],'가설 2  거래가 많은 단지는 덜 떨어지고 더 회복했을까?','01_가설2_거래활발도','분기당 거래')
    t31,_=group_plot(d,'households',['작은 단지','중간','큰 단지'],'가설 3-1  같은 구에서 큰 단지가 더 잘 버텼을까?','02_가설3-1_세대수','세대수')
    # Negative age sorts oldest first; show actual age in the explanatory labels/table.
    d['newness']=-d.age
    t32,g32=group_plot(d,'newness',['오래된 편','중간','연식 짧은 편'],'가설 3-2  같은 구에서 연식이 짧은 단지가 더 잘 버텼을까?','03_가설3-2_연식','연식')
    t32['특성중앙값']=-t32['특성중앙값'];export(t32,'03_가설3-2_연식_숫자')
    l=location(d);p,details,n,c=pairs(d)
    report(d,meta,[t2,t31,t32],l,p,details,n,c)
    print('Pair summary:',n,c,len(p),p['낙폭차이(%p)'].median(),p['회복률차이(%p)'].median(),flush=True)

if __name__=='__main__':main()
