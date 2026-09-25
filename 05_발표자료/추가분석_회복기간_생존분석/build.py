"""Figures and narrative aligned to the newly supplied original sections 4–6."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
OLD=ROOT/'05_발표자료/발표_EDA_20260908'
FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
facts=json.loads((OLD/'findings.json').read_text(encoding='utf-8'))
source=ROOT/'거래_단지_DATA/아파트_거래_DATA_대시보드_실데이터.csv'
with source.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
assert actual==facts['quality']['sha256'],'Source changed; recompute base EDA before using it.'
p=pd.read_csv(OLD/'03_EDA_동일시점_하락관측_표본.csv')
p=p.loc[p.area_group.eq('60㎡ 이상 85㎡ 미만')].copy()
q=pd.read_csv(OLD/'04_단지면적군_분기집계.csv')
live=pd.read_csv(OLD/'02_전체조합_대시보드잠정_지표.csv')
sizes=pd.read_csv(OLD/'11_규모별_60_85_요약.csv')
p['event']=p.recovery_quarters.notna()
p['duration']=p.recovery_quarters.fillna(p.followup_quarters).astype(int)
assert p.duration.ge(0).all() and p.loc[p.event,'duration'].gt(0).all()
assert len(p)==894 and p.complex_id.nunique()==894

plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':12,
 'figure.facecolor':'white','axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,
 'axes.titleweight':'bold','axes.titlesize':14,'text.color':'#14263d','axes.labelcolor':'#334155',
 'axes.grid':True,'grid.color':'#e2e8f0','axes.axisbelow':True,'svg.fonttype':'path'})
blue,orange,green='#2563eb','#d95f02','#00856a'
def save(fig,name,note):
    fig.text(.06,.022,note,fontsize=10,color='#475569')
    fig.tight_layout(rect=[.015,.075,.99,.94])
    fig.savefig(FIG/f'{name}.png',dpi=180)
    fig.savefig(FIG/f'{name}.svg')
    plt.close(fig)
def selected(df,names,gu):
    return [df.loc[df['name'].eq(n)&df.district.eq(gu)&df.area_group.eq('60㎡ 이상 85㎡ 미만')].iloc[0] for n in names]
song=selected(p,['리센츠','헬리오시티','파크리오'],'송파구')
now=selected(p,['보람아파트','건영3','한진한화그랑빌'],'노원구')
periods=pd.period_range('2020Q1','2026Q2',freq='Q').astype(str)
fig,axes=plt.subplots(1,2,figsize=(13.4,7.5))
for r,color in zip(song,[blue,orange,green]):
    s=q.loc[q.complex_id.eq(r.complex_id)&q.area_group.eq(r.area_group)&q['count'].ge(3)].set_index('quarter').price.reindex(periods)
    for ax,scale in zip(axes,[1,100/r.peak]):
        ax.plot(range(len(s)),s*scale,label=r['name'],color=color,lw=2.2,marker='o',ms=3)
        for time,val,marker in [(r.peak_q,r.peak,'^'),(r.trough_q,r.trough,'v')]:
            ax.scatter(list(periods).index(time),val*scale,s=95,marker=marker,color=color,zorder=5)
for ax in axes:
    ticks=[0,4,8,12,16,20,25]
    ax.set_xticks(ticks,[periods[i] for i in ticks],rotation=35,ha='right');ax.legend(fontsize=10)
axes[0].set(title='분기별 ㎡당 중위 거래가격',ylabel='만원/㎡')
axes[1].set(title='MDD 기준 고점=100으로 비교',ylabel='가격지수');axes[1].axhline(100,color='#64748b',ls='--')
fig.suptitle('4-1. 같은 송파구·면적군에서도 하락과 회복의 크기는 다르다',fontsize=18,fontweight='bold')
save(fig,'01_고점_저점_현재','60~85㎡ · ▲고점 2021Q4, ▼저점 2022Q4 · 지표는 2017Q1부터 계산 · 3건 미만 분기는 공백 · 가격축 0 생략')

fig,ax=plt.subplots(figsize=(13.4,7.5))
ax.scatter(p.mdd,p.recovery,s=20,alpha=.38,color='#94a3b8',label=f'동일 면적군 {len(p):,}개 단지')
for r,color in zip(now,[orange,blue,green]):
    ax.scatter(r.mdd,r.recovery,s=110,color=color,edgecolors='white',zorder=5,label=f'{r["name"]} · {r.recovery:.1f}%')
ax.axhline(100,color='#64748b',ls='--',label='기준 고점 회복 100%')
ax.set_yscale('symlog',linthresh=100);ax.set_ylim(-5,p.recovery.max()*1.5)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:,.0f}'))
ax.set(xlabel='MDD (%)   ← 더 큰 하락                  더 작은 하락 →',ylabel='하락회복률 (%) · 100% 초과 구간 로그 눈금')
ax.legend(loc='upper left',fontsize=10)
rho=p.loss.rank().corr(p.recovery.rank())
fig.suptitle('4-2. 낙폭과 회복률은 분산되어 있고, 같은 낙폭도 다른 결과로 이어진다',fontsize=17,fontweight='bold')
save(fig,'02_MDD_회복률_산점도',f'2026Q2 · 60~85㎡ · 하락 관측 894개 단지 · 낙폭 크기 |MDD|와 회복률의 순위상관 ρ={rho:.3f} · 극단값 포함')

def km(g):
    survival=1.; rows=[]
    for time in sorted(g.duration.unique()):
        risk=int(g.duration.ge(time).sum())
        deaths=int((g.duration.eq(time)&g.event).sum())
        censor=int((g.duration.eq(time)&~g.event).sum())
        survival*=1-deaths/risk
        rows.append(dict(quarter=int(time),at_risk=risk,events=deaths,censored=censor,
                         recovery_pct=(1-survival)*100))
    frame=pd.DataFrame(rows)
    assert frame.events.sum()==int(g.event.sum())
    assert frame.censored.sum()==int((~g.event).sum())
    assert frame.recovery_pct.diff().dropna().ge(-1e-9).all()
    assert frame.recovery_pct.between(0,100.00000001).all()
    return frame
def at(curve,t):
    a=curve.loc[curve.quarter.le(t),'recovery_pct']
    return float(a.iloc[-1]) if len(a) else 0.
curves=[]; speed=[]
fig,(ax,rx)=plt.subplots(2,1,figsize=(13.4,7.5),gridspec_kw={'height_ratios':[4,1]})
rx.axis('off')
risk_rows=[]
for label,mask,color in [('낙폭 20% 미만',p.loss.lt(20),blue),('낙폭 20% 이상',p.loss.ge(20),orange)]:
    g=p.loc[mask]; cv=km(g);cv['group']=label;curves.append(cv)
    plotcv=cv.loc[cv.quarter.le(12)]
    xx=[0]+plotcv.quarter.tolist()+[12];yy=[0]+plotcv.recovery_pct.tolist()+[at(cv,12)]
    ax.step(xx,yy,where='post',color=color,lw=2.5,label=f'{label} (n={len(g)})')
    ax.scatter(8,at(cv,8),color=color,s=65)
    ax.annotate(f'8분기 {at(cv,8):.1f}%',(8,at(cv,8)),xytext=(14,7),textcoords='offset points',color=color,fontweight='bold')
    med=cv.loc[cv.recovery_pct.ge(50),'quarter']
    speed.append(dict(group=label,n=len(g),recovered=int(g.event.sum()),unrecovered=int((~g.event).sum()),
        km_4q=at(cv,4),km_8q=at(cv,8),km_12q=at(cv,12),median_q=int(med.iloc[0]) if len(med) else None,
        at_risk_8q=int(g.duration.ge(8).sum())))
    risk_rows.append([label]+[str(int(g.duration.ge(t).sum())) for t in [0,4,8,12]])
ax.set(xlim=(0,12.3),ylim=(0,100),xlabel='각 단지의 MDD 저점 이후 경과 분기',ylabel='최초 고점 회복 누적 추정 비율 (%)')
ax.set_xticks(range(0,13,2));ax.legend(loc='upper left',fontsize=11)
tab=rx.table(cellText=risk_rows,colLabels=['미회복 추적 대상 수*','0분기','4분기','8분기','12분기'],cellLoc='center',loc='center',colWidths=[.36,.16,.16,.16,.16])
tab.auto_set_font_size(False);tab.set_fontsize(11);tab.scale(1,1.6)
fig.suptitle('4-3. 회복 속도는 회복률 대신 최초 고점 회복까지의 시간으로 확인한다',fontsize=17,fontweight='bold')
save(fig,'03_가설1_최초회복_기간','Kaplan–Meier 탐색 추정 · 미회복은 Q2에서 우측 검열 · *해당 분기 직전 위험집단 · 저점 연도·관측 공백 차이 존재')
pd.concat(curves).to_csv(OUT/'회복기간_KM_위험집단_계산표.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(speed).to_csv(OUT/'회복기간_집단별_요약.csv',index=False,encoding='utf-8-sig')
sens=[]
for threshold in [10,15,20,25]:
    for low in [True,False]:
        g=p.loc[p.loss.lt(threshold) if low else p.loss.ge(threshold)]
        cv=km(g)
        sens.append({'threshold':threshold,'group':'미만' if low else '이상','n':len(g),
                     'km_8q':at(cv,8),'at_risk_8q':int(g.duration.ge(8).sum())})
pd.DataFrame(sens).to_csv(OUT/'회복기간_낙폭구분_민감도.csv',index=False,encoding='utf-8-sig')
fixed=p.loc[p.followup_quarters.ge(8)].assign(group=lambda x:x.loss.ge(20),recovered8=lambda x:x.recovery_quarters.le(8))
fixed_summary=fixed.groupby('group').agg(n=('complex_id','size'),recovered8=('recovered8','sum'))
fixed_summary['rate']=fixed_summary.recovered8/fixed_summary.n*100
fixed_summary.to_csv(OUT/'회복기간_8분기추적_민감도.csv',encoding='utf-8-sig')

fig,(ax,bx)=plt.subplots(1,2,figsize=(13.4,7.5))
names=['보람아파트\n상계동','건영3\n중계동','한진한화그랑빌\n월계동']
for axis,col,title in [(ax,'loss','최대 하락폭의 크기'),(bx,'recovery','하락분 회복 수준')]:
    values=[r[col] for r in now]
    bars=axis.bar(names,values,color=[orange,blue,green],width=.6)
    axis.bar_label(bars,labels=[f'{v:.1f}%' for v in values],padding=5,fontweight='bold')
    axis.set(title=title,ylabel='%',ylim=(0,max(values)*1.3))
bx.axhline(100,color='#64748b',ls='--',label='고점 회복 100%');bx.legend(loc='upper left',fontsize=10)
fig.suptitle('4-4. 노원구·60~85㎡: 약 30% 하락한 두 단지의 회복률은 42.7%와 134.1%',fontsize=17,fontweight='bold')
save(fig,'04_가설2_동일지역_단지비교','2026Q2 · 분기 거래 보람 28건 / 건영3 14건 / 한진한화 30건 · 저점 시기·동·연식 등은 동일하지 않음 · 대표성 대신 대비 사례')

fig,axes=plt.subplots(1,3,figsize=(13.4,7.5))
labels=['300세대\n미만','300~999\n세대','1,000세대\n이상']
for axis,col,title,unit in [(axes[0],'mdd_median','낙폭 크기 중앙값','%'),(axes[1],'recovery_median','회복률 중앙값','%'),(axes[2],'qualified_q_median','유효 분기 수 중앙값','개')]:
    vals=sizes[col].to_numpy(); vals=-vals if col=='mdd_median' else vals
    bars=axis.bar(labels,vals,color=[blue,'#648ee9','#b0c5f4'],width=.6)
    texts=[f'{v:.1f}{unit}' for v in vals]
    axis.bar_label(bars,labels=texts,padding=5)
    axis.set(title=title,ylim=(0,max(vals)*1.3),ylabel='낙폭 크기 (%)' if col=='mdd_median' else unit)
fig.suptitle('4-5. 대단지는 더 크게 하락했지만, 하락을 관측할 거래 이력도 더 많았다',fontsize=17,fontweight='bold')
save(fig,'05_가설3_규모와_관측이력','60~85㎡ · 세대수 확인 875개: 158 / 473 / 244개 · 보정 전 세대수↔낙폭 ρ=0.495, 구·연식·관측 분기 보정 후 0.096')

fig,(ax,bx)=plt.subplots(1,2,figsize=(13.4,7.5))
counts=[5897,1012,894,875]
labels=['60~85㎡ 전체 관측 단지','Q2 동일시점 지표 가능','그중 하락 관측','그중 세대수 확인']
bars=ax.barh(labels,counts,color=['#cbd5e1','#8eb3f4',blue,green]);ax.invert_yaxis()
ax.bar_label(bars,labels=[f'{v:,}' for v in counts],padding=5);ax.set_xlim(0,7000)
ax.set(title='동일 면적군의 분석 가능 범위',xlabel='단지 수')
bars=bx.bar(['분기 거래 1~2건','분기 거래 3건 이상'],[122669,66121],color=[orange,blue])
bx.bar_label(bars,labels=['122,669 (65.0%)','66,121 (35.0%)'],padding=7);bx.set_ylim(0,145000)
bx.set(title='분기별 거래 표본의 크기',ylabel='전체 면적군의 관측 분기 셀 수')
fig.suptitle('6. 많은 거래 데이터가 모든 단지의 충분한 관측을 의미하지는 않는다',fontsize=17,fontweight='bold')
save(fig,'06_표본한계와_개선필요','왼쪽 분모: Q3까지 CSV 전체 관측 단지 · 분석은 Q2 기준 · 오른쪽: 전체 면적군·전체 기간, 거래 없는 셀은 제외')

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])
songtable=table(['단지','고점 가격','저점 가격','Q2 가격','MDD','회복률','Q2 거래'],
 [[r['name'],f'{r.peak:,.1f}',f'{r.trough:,.1f}',f'{r.price:,.1f}',f'{r.mdd:.1f}%',f'{r.recovery:.1f}%',r.basis_count] for r in song])
speedtable=table(['낙폭 구분','표본','회복 관측 / 미회복','8분기 내 최초 회복 추정','회복기간 중앙값 추정'],
 [[s['group'],s['n'],f'{s["recovered"]} / {s["unrecovered"]}',f'{s["km_8q"]:.1f}%',f'{s["median_q"]}분기'] for s in speed])
livet=selected(live,['리센츠','헬리오시티','파크리오'],'송파구')+selected(live,['보람아파트','건영3','한진한화그랑빌'],'노원구')
livetable=table(['구','단지','Q3 잠정 가격','MDD','회복률','고점 대비 위치','Q3 거래'],
 [[r.district,r['name'],f'{r.price:,.1f}',f'{r.mdd:.1f}%',f'{r.recovery:.1f}%',f'{r.gap:+.1f}%',r.basis_count] for r in livet])
nowtable=table(['노원구 단지','Q2 가격','MDD','회복률','고점 대비 위치'],
 [[r['name'],f'{r.price:,.1f}',f'{r.mdd:.1f}%',f'{r.recovery:.1f}%',f'{r.gap:+.1f}%'] for r in now])
sizetable=table(['세대수 구간','표본','MDD 중앙값','회복률 중앙값','유효 분기 중앙값'],
 [[r.size_group,r.n,f'{r.mdd_median:.1f}%',f'{r.recovery_median:.1f}%',f'{r.qualified_q_median:.0f}개'] for r in sizes.itertuples()])
thin=p.loc[p.loss.ge(10)]
stats={'source_sha256':actual,'area':'60㎡ 이상 85㎡ 미만','n':len(p),'mdd_median':p.mdd.median(),
       'recovery_median':p.recovery.median(),'rho_loss_recovery':rho,
       'rho_loss_ge_10':thin.loss.rank().corr(thin.recovery.rank()),'n_loss_ge_10':len(thin),
       'speed':speed,'sensitivity':sens}
(OUT/'분석요약.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
p.to_csv(OUT/'EDA_60-85_분석표본.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(song+now+livet).to_csv(OUT/'시연단지_Q2_Q3_근거.csv',index=False,encoding='utf-8-sig')

doc=f'''# 원문 4~6번에 맞춘 EDA 시각자료와 발표 구성

기준 문서: [서울_아파트_가격_회복탄력성_원문.md](../../서울_아파트_가격_회복탄력성_원문.md)  
분석 원천: [지정 실데이터 CSV](../../거래_단지_DATA/아파트_거래_DATA_대시보드_실데이터.csv)  
작성일: 2026-09-08. 원문과 원본 CSV, 대시보드 코드는 수정하지 않았다.

**발표의 중심 메시지: 같은 구·면적군에서도 하락폭과 회복 수준은 다르므로 단지별로 비교해야 한다. 다만 회복 속도와 표본 신뢰도까지 구분해서 해석해야 한다.**

## 4. EDA — 실제로 단지별 차이가 존재하는가?

### 분석 기준

- 데이터: 2017-01-01~2026-08-31, 610,341건, 서울 25개 구·8,963개 단지.
- EDA 기준 시점: **2026년 2분기**. 미완결 3분기는 제외한다. 데이터 범위상 완결이라는 뜻이며 신고 지연·추후 정정까지 완료됐다는 의미는 아니다.
- 비교 면적: **60㎡ 이상 85㎡ 미만**으로 통일한다. 이 면적군은 전체 관측 단지 5,897개 중 Q2 동시점 지표 가능 1,012개, 그중 하락 관측 **894개 단지**다. 규모 분석은 세대수가 확인된 875개다. 전체 관측 분모에는 Q3까지 등장한 단지를 포함한다.
- 가격: 개별 거래의 `거래금액(만원) / 면적(㎡)`을 계산한 뒤 단지·면적군·분기별 **중앙값**을 사용한다.
- 표본 기준: 분기 거래 3건 이상, 그러한 완결 분기 4개 이상, Q2에도 거래 3건 이상. 4개는 연속 분기라는 뜻은 아니다.
- MDD는 누적 고점 대비 하락률의 최솟값이다. 그 최대 하락 구간의 고점 H·저점 L을 기준으로 `회복률=(Q2 가격−L)/(H−L)×100`을 계산한다.
- **H는 MDD 구간의 기준 고점이며 이후 갱신된 사상 최고가는 아니다.** 하락이 관측되지 않은 118개 단지는 MDD=0, 회복률은 미정의이므로 회복률·회복기간 분석에서 제외했다.
- 거래 없는 분기는 채우지 않는다. 3건 미만 분기도 지표에서 제외하며 가격 그래프에서 공백을 연결하지 않는다.

이전 산출물과 원천 CSV의 SHA-256이 일치함을 확인하여 검증된 분기 집계를 재사용했다. 이번에는 전체 면적을 섞지 않고 동일 면적군 894개로 범위를 통일하고, 원문의 ‘빠를까’에 답하기 위한 회복기간 분석을 추가했다.

### 4-1. 고점 → 저점 → 현재: 실제 가격 흐름부터 보여주기

![고점·저점·현재 가격](figures/01_고점_저점_현재.png)

**송파구 리센츠·헬리오시티·파크리오**는 MDD 기준 고점이 모두 2021Q4, 저점이 모두 2022Q4다. 같은 시기에 하락과 회복을 겪은 단지들이어서 시작 사례로 적합하다. 가격 단위는 만원/㎡다.

{songtable}

왼쪽은 실제 가격 수준, 오른쪽은 단지별 고점=100인 가격지수다. **오른쪽 지수 130.7은 고점 대비 가격 +30.7%라는 뜻이며 하락회복률 130.7%라는 뜻은 아니다.**

발표 멘트:

> “같은 송파구, 같은 면적군에서 세 단지는 같은 시기에 고점과 저점을 경험했습니다. 하지만 최대 하락폭과 이후 회복 수준은 같지 않았습니다. 현재 가격뿐 아니라 그 가격에 도달한 과정도 비교할 필요가 있습니다.”

### 4-2. 산점도: 단지별 분포와 MDD·회복률 관계

![MDD·회복률 산점도](figures/02_MDD_회복률_산점도.png)

- 점 1개는 60~85㎡ 단지 1개이며, 전체 **894개**를 표시했다.
- X축은 MDD로 오른쪽일수록 하락폭이 작다. Y축은 하락회복률로 100%가 기준 고점 회복이다.
- 이 표본의 MDD 중앙값은 **{p.mdd.median():.1f}%**, 회복률 중앙값은 **{p.recovery.median():.1f}%**다. 전체 면적군 2,014개에서 계산한 −15.7%·162.5%와 분모가 다르다.
- 낙폭 크기 |MDD|와 회복률의 Spearman 순위상관은 **{rho:.3f}**다. 즉 작은 낙폭과 높은 회복률이 함께 관측되는 경향이 있다. X축의 음수 MDD 자체로 계산하면 상관 부호는 반대다.
- 낙폭 10% 이상 {len(thin)}개로 제한하면 상관은 **{stats['rho_loss_ge_10']:.3f}**다. 작은 낙폭에서 회복률 분모가 작아지는 현상을 함께 고려해야 한다.

회복률 최댓값은 이 면적군에서도 약 2,718,940%다. 용산구 벽산메가트리움의 관측 MDD가 약 −0.005884%로 극히 작아 발생한다. 극단값은 삭제하지 않았고, 그림은 100% 이하 선형·그 이상 로그 눈금으로 표시했다. **회복률이 높다는 사실만으로 큰 폭 상승이나 빠른 회복을 뜻하지 않는다.**

### 4-3. 가설 1 — “하락폭이 작은 단지가 회복도 빠를까?”

회복률은 ‘수준’이므로 속도는 **MDD 저점 이후 처음 기준 고점에 도달한 분기까지의 기간**으로 별도 계산했다. 미회복 단지를 분석에서 버리지 않고 Q2까지 회복이 관측되지 않은 상태로 처리하는 Kaplan–Meier 방법을 사용했다.

![회복기간 탐색 분석](figures/03_가설1_최초회복_기간.png)

{speedtable}

곡선은 ‘현재 고점 이상인 단지 비율’이 아니라 **저점 이후 최초 고점 회복이 발생한 누적 비율의 탐색 추정치**다. 회복한 뒤 다시 하락한 단지도 최초 회복 사건에 포함된다. 표의 ‘미회복 추적 대상 수’는 해당 경과 분기 직전까지 사건이나 추적 종료가 발생하지 않은 단지 수다.

**해석:** 이 표본에서는 20% 미만 하락 집단이 최초 고점 회복에 더 빨리 도달하는 패턴을 보인다. 그러나 ‘20%’는 탐색을 위한 구분이며 객관적으로 검증된 안전 기준은 아니다. 10·15·25% 구분의 민감도는 [계산표](회복기간_낙폭구분_민감도.csv)에 수록했다.

보조 점검으로 저점 이후 **최소 8분기의 경과 기간이 확보된 단지**만 보면, 8분기 내 회복 관측 비율은 작은 낙폭 집단 258/361개(71.5%), 큰 낙폭 집단 55/256개(21.5%)다. 이는 분기마다 거래가 있었다는 뜻은 아니며, 최신 저점 단지가 제외되는 다른 표본이다. 계산표는 [8분기 추적 민감도](회복기간_8분기추적_민감도.csv)에 있다.

**검증 범위:** 미회복은 2026Q2에 우측 검열했고, 최초 회복은 거래 3건 이상 분기에서 관측된 때다. 실제 회복 시점은 거래 공백 사이에 있었을 수 있다. 저점 연도·상승장 노출·연식·관측 밀도가 다르고 MDD 구간도 전체 기간을 본 뒤 선택한 것이므로 인과 효과나 미래 예측으로 해석하지 않는다. Kaplan–Meier의 비정보성 검열 가정, 즉 추적 종료 시점이 회복 가능성과 체계적으로 연관되지 않는다는 가정은 검증하지 않았다. 유의성 검정·신뢰구간도 이번 탐색에 포함하지 않았다. 특히 낙폭이 작으면 회복해야 하는 가격 차이 자체도 작다.

발표 멘트:

> “회복률과 회복 속도를 구분했습니다. 최초 고점 회복까지의 기간을 보면 작은 낙폭 집단이 더 빨리 회복하는 패턴이 나타납니다. 다만 회복해야 할 폭과 저점 시기가 다르기 때문에 이를 단지의 고유한 능력이나 인과관계로 단정하지 않았습니다.”

### 4-4. 가설 2 — “같은 자치구·면적군에서도 단지별 차이가 있을까?”

![노원구 동일 면적군 비교](figures/04_가설2_동일지역_단지비교.png)

{nowtable}

**보람아파트와 한진한화그랑빌**은 MDD가 각각 −29.5%, −29.7%로 비슷하지만, Q2 회복률은 **42.7%와 134.1%**로 다르다. 기준 고점은 둘 다 2021Q3이고 저점은 각각 2024Q1, 2023Q1이다. 회복에 주어진 시간도 다르므로 그 차이까지 가격 추이에서 읽는다.

건영3은 기준 고점 대비 +0.9%다. 세 단지를 보면 **미회복 / 고점 부근 / 고점 초과** 상태가 같은 구·면적군 안에 공존한다. 층·연식·동·정확한 면적까지 같은 비교는 아니며 사례의 차이를 전체 단지의 평균적인 차이로 일반화하지 않는다.

발표 멘트:

> “같은 노원구·같은 면적군에서 약 30% 하락했던 두 단지도 회복 수준은 달랐습니다. 자치구 평균만으로는 관심 단지의 상태를 설명하기 어려워 단지별 비교가 필요합니다.”

### 4-5. 가설 3 — “대규모와 소규모 단지는 다를까?”

![규모와 관측 이력](figures/05_가설3_규모와_관측이력.png)

{sizetable}

대규모에서 더 큰 낙폭이 관측되지만 유효 분기 수도 많다. 세대수와 낙폭 크기의 단순 순위상관은 **0.495**지만 자치구·건축연도·유효 분기 수를 보정하면 **0.096**이다. 세대수와 회복률의 보정 후 관계도 **0.008**로 작다. 부분 순위상관은 순위값을 지역 더미·연식 순위·관측 분기 순위에 회귀한 잔차 사이 상관으로 계산했다.

**결론:** 규모별 관측 차이는 있으나 대단지가 더 안전하거나 더 잘 회복한다고 단정하기 어렵다. 거래가 많은 단지는 하락을 포착할 기회도 많다. 규모 구분은 EDA용이며 제도상 분류가 아니다. 관측 밀도 보정이 규모의 인과 효과를 추정한 것은 아니다.

발표 멘트:

> “대단지는 더 큰 낙폭이 관측됐지만, 가격을 관찰할 수 있는 분기도 더 많았습니다. 지역·연식·관측 분기를 고려하면 규모와 지표의 관계가 작아져 세대수만으로 회복력을 판단하기 어렵습니다.”

### 발표 흐름 추천

그림 01(가격 과정) → 그림 02(분포) → 그림 03(속도 가설) → 그림 04(같은 조건에서도 차이) → 그림 05(규모만으로 설명 불가) → **5번 시연** → 그림 06과 **6번 한계·개선** 순서가 원문과 맞는다. 시간이 짧으면 그림 02·03·05는 보충 자료로 두고 01→04→시연을 중심으로 발표한다.

## 5. 대시보드 — EDA와 연결되는 시연 시나리오

### 먼저 맞춰야 하는 데이터 기준

실제 데이터 발표이므로 앱의 **‘실데이터’** 모드를 사용한다. **‘시연 화면’은 고정 예시 데이터**이므로 이번 EDA의 근거로 쓰지 않는다. EDA는 2026Q2이고 현재 앱은 2026Q3(7~8월) 잠정값을 쓴다. 앱에 분기 기준 선택 입력은 현재 없으므로 “앞의 분석은 Q2, 지금 화면은 Q3 잠정값”이라고 설명한다.

기본 입력은 `.csv.gz`이므로 이번 분석과 같은 `.csv`를 명시하려면 프로젝트 루트에서 다음처럼 실행한다.

```powershell
$env:APARTMENT_DATA_CSV = (Resolve-Path -LiteralPath '거래_단지_DATA/아파트_거래_DATA_대시보드_실데이터.csv').Path
python -m streamlit run '대시보드/streamlit_app.py' --browser.gatherUsageStats false
```

아래 6개 후보는 모두 Q3에 거래 3건 이상이므로 같은 잠정 분기끼리 비교된다. 가격은 만원/㎡다.

{livetable}

앱의 **‘단지 분포’는 MDD × 고점 대비 위치**이고, 회복률은 **‘하락분 회복률과 산출 기준 보기’**에 있다. 이번 그림 02와 앱 분포 차트의 Y축이 다름을 설명한다. 특정 면적으로 환산한 화면 총액도 실제 한 건의 거래금액과 구분한다.

### 추천 1. 노원구에서 비슷하게 하락한 후보들의 회복 상태 비교 — 분석 메시지 강조

**연결:** 그림 04 → 시연. 원문의 “실제로 차이가 있는가?”를 가장 명확하게 보여주는 시나리오다. 약 90초.

1. **조건 선택:** 실데이터 → 노원구 → 60㎡ 이상 85㎡ 미만.
2. **단지 확인:** 보람아파트 · 상계동 → ‘결과 보기’.
3. **가격 흐름 확인:** 최대 하락폭 −29.5%, 현재 MDD 기준 고점 대비 −11.3%, Q3 표본 3건을 확인한다.
4. **유사 단지 비교:** 추가 비교에 건영3 · 중계동, 한진한화그랑빌 · 월계동을 선택한다.
5. 고점 대비 위치 −11.3% / +4.5% / +18.1%를 비교한다. 상세 회복률은 각 단지를 기준 단지로 바꾸고 ‘결과 보기’를 적용한 뒤 펼쳐서 확인한다.
6. 보람아파트와 한진한화그랑빌의 비슷한 MDD(−29.5%/−29.7%)와 다른 회복률(61.8%/161.1%)을 강조한다.

대사: “하락폭이 비슷했던 단지라도 지금 회복한 수준은 다릅니다. 원하는 조건 안에서 개별 단지의 과정과 현재 위치를 함께 확인할 수 있습니다.”

보람아파트 Q3는 3건이므로 변화가 확정됐다고 말하지 않는다. Q2의 더 많은 표본(28건)을 EDA 근거로 유지한다.

### 추천 2. 송파구에서 리센츠·헬리오시티·파크리오 비교 — 자연스러운 사용 흐름

**연결:** 그림 01 → 시연. 기본 선택 단지와 같아 조작이 간단하다. 약 90초.

1. 실데이터 → 송파구 → 60㎡ 이상 85㎡ 미만 → 리센츠 · 잠실동 → ‘결과 보기’.
2. 가격 추이에서 고점 2021Q4, 저점 2022Q4, 현재 Q3 잠정 가격을 짚는다.
3. 추가 비교로 헬리오시티 · 가락동, 파크리오 · 신천동을 선택한다.
4. 세 카드의 MDD와 고점 대비 현재 위치를 비교한다.
5. 리센츠의 상세 회복률 270.4%를 열고 “가격이 270.4% 올랐다는 뜻이 아니라 과거 하락분의 2.704배를 회복했다는 뜻”이라고 설명한다.
6. ‘단지 분포’에서 같은 구·면적군의 상대 위치를 보여준다.

대사: “현재 가격뿐 아니라 과거에 얼마나 떨어졌고 어느 정도 회복했는지를 같은 조건으로 비교할 수 있습니다.”

### 추천 3. 송파더플래티넘의 ‘자료 부족’ 확인 — 6번 한계점 연결

**연결:** 시연 마지막 → 6번. 약 30~45초이며 추천 1 또는 2 뒤에 붙인다.

1. 송파구 → 60㎡ 이상 85㎡ 미만 → 송파더플래티넘 · 오금동 → ‘결과 보기’.
2. 이 면적군의 CSV 거래는 11건, 거래 관측 분기는 5개다. Q3에는 3건이 있어 분기 가격 약 2,096.4만원/㎡는 확인할 수 있다.
3. 하지만 3건 이상인 **완결 분기는 1개뿐**이므로 MDD·회복률의 4분기 조건을 충족하지 못한다.

대사: “가격이 확인된다고 장기 회복력까지 판단할 수 있는 것은 아닙니다. 자료 부족을 회복 실패나 하락 없음으로 처리하지 않았습니다.”

추천 조합은 **노원구 비교 + 자료 부족 사례**다. EDA의 발견을 강조하면서 한계점까지 자연스럽게 이어진다. 매수 추천이나 미래 수익 전망으로 마무리하지 않고 ‘비교에 필요한 정보를 제공한다’는 서비스 목적에 맞춘다. 이번 작업은 시나리오와 계산값 준비까지이며 새 브라우저 시연·영상 녹화는 수행하지 않았다.

## 6. 한계점 및 고도화 방향

![표본 한계](figures/06_표본한계와_개선필요.png)

### 현재 한계와 대응 방향

| 한계 | 데이터 근거·해석 범위 | 고도화 방향 |
|---|---|---|
| 거래가 얕은 분기 | 전체 면적군의 관측 분기 셀 188,790개 중 122,669개(65.0%)가 1~2건. 거래 없는 분기는 이 분모에도 없음 | 거래 수·유효 분기 수·최근 관측일을 묶은 신뢰도 안내, 분기 최소 3·5·10건 민감도 제공 |
| 분석 가능 단지에 편중 | 60~85㎡ 전체 관측 5,897개 중 Q2 동시점 지표 가능은 1,012개(17.2%). 회복률 표본은 894개 | 구·면적별 분석 가능률, 제외 이유, 거래 많은/적은 단지의 특성 차이 공개 |
| 거래 공백과 가격 구성 변화 | 유효 분기만 사용하므로 고점·저점을 놓칠 수 있음. 같은 면적군이라도 정확한 면적·층·연식이 다름 | 상세 면적·층·연식 보정, 동일 관측창 비교, 반복매매·헤도닉 가격의 민감도 분석 |
| 미완결 분기와 서로 다른 기준일 | Q3는 7~8월 잠정. 전체 면적군 앱 지표 가능 4,435개 중 3,700개(83.4%)는 더 과거의 유효 분기를 사용 | 완결/잠정 전환, 비교 기준 분기 고정, 오래된 가격의 경과 분기 표시 |
| 회복률의 작은 분모 | 낙폭이 0에 가까우면 회복률이 매우 커짐 | 원값은 유지하면서 얕은 낙폭 표시, 중앙값·MDD·고점 대비 위치 병기 |
| 규모·입지 결측과 매칭 | 전체 8,963단지 중 세대수 미확인 5,998개(66.9%). 단지·입지 정보 묶음 차이 312개 | 미매칭 단지 보완, 매칭 등급·변경 이력 관리, 규모 결측 집단과 관측 집단 비교 |
| 입지는 거래 당시와 다를 수 있음 | 최근접 시설의 직선거리이며 과거 개통일·보행거리·학교 배정 정보와 다름 | 거래 당시 입지 정보와 연결, 도보 접근성 개선, 시설 미매핑과 실제 부재 구분 |
| 회복 기간도 완전히 관측되지 않음 | 거래 공백 사이에 실제 회복했을 수 있고, 단지마다 저점 연도가 다름. MDD 구간은 사후 선택됨 | 고점·하락 국면을 사전에 정의한 분석, 구간 검열 검토, 저점 시기·관측 빈도 보정, 생존곡선 신뢰구간 |
| 상관을 원인으로 해석할 수 없음 | 낙폭과 회복률은 H·L을 공유하고, 규모·입지 외에 지역·연식·시장 상황이 작용 | 관측창·주택 특성을 보정한 설명모형과 시점별 검증. 상관·예측 기여도와 인과 효과 구분 |

거래 수·자료 부족·잠정값·거래 공백 표시는 앱에 이미 있다. **새로 추가할 부분은 동일 시점 선택, 신뢰도 수준의 구체화, 관측창 및 구성 보정, 요인 설명**이다. 이미 있는 기능을 향후 새 기능으로 소개하지 않는다.

### 고도화 순서

1. **비교의 신뢰도 강화:** 원천 버전·기준일·단지 식별 이력을 남기고, 같은 분기 비교와 신뢰도 안내를 제공한다. 표본 기준을 높이면 결과뿐 아니라 제외 단지 수가 어떻게 바뀌는지도 함께 보여준다.
2. **회복 기간·영향요인 설명:** 이번 회복기간 탐색을 확장해 미회복과 거래 공백을 반영한다. 세대수·연식·입지와 거래 당시 금리·대출규제 정보를 연결하고 지역·시간·관측량을 고려한 모형으로 관계를 분석한다. 모든 단지에 공통인 금리와 완전한 분기 효과는 중복될 수 있어 별도 식별 설계가 필요하다.
3. **설명 가능한 비교 서비스:** 비슷한 가격대·면적·연식의 후보를 찾고, 어떤 특성과 표본 때문에 현재 결과가 나왔는지 설명한다. 현재의 방어/회복 가중치 20:80은 설계 선택이므로 가중치 변화에 따른 순위 민감도도 제시한다.

완료 여부는 같은 분기 값의 실제 일치, 결측·잠정·과거 기준값의 구분, 시점을 나눈 검증 성능, 구·면적별 결과 안정성으로 평가한다. 외부 요인 분석은 향후 제안이며 이번 작업에서 해당 데이터를 추가하거나 영향력을 검증한 것은 아니다.

발표용 마무리:

> “현재 프로젝트는 단지별 가격 회복력을 비교하는 단계입니다. 실거래 기록은 단지마다 관측량과 시점이 다르기 때문에 수치와 함께 신뢰도를 보여줘야 합니다. 앞으로는 같은 시점의 비교와 가격 구성 보정을 강화하고, 회복 기간 및 금리·입지 등의 관계까지 분석해 결과의 이유를 설명하는 서비스로 발전시키겠습니다.”

## 자료 사용과 검증

- 그림 6종은 `figures/`에 PNG(2,412×1,350)와 SVG로 제공한다. PNG는 슬라이드에 삽입하고 SVG는 확대·편집에 사용할 수 있다.
- [894개 단지 분석표본](EDA_60-85_분석표본.csv), [시연 단지 Q2·Q3 근거](시연단지_Q2_Q3_근거.csv), [회복기간 위험집단 계산표](회복기간_KM_위험집단_계산표.csv), [집단별 회복기간 요약](회복기간_집단별_요약.csv).
- 원천 SHA-256: `{actual}`. 이전 검증에서 원천 거래 합계와 분기 집계를 대조했고, Q2/Q3 90개 표본과 시연 6개 단지·자료 부족 1개 단지를 앱 계산 함수와 대조했다.
- 이번 추가 회복기간 계산은 집단별 사건·검열 합계가 원표본과 일치하는지, 누적 회복 추정치가 0~100% 범위에서 감소하지 않는지 점검했다. 표본 평균 회복기간에서 미회복 단지를 제외하는 방식은 사용하지 않았다.
'''
(OUT/'원문_4-6_EDA_시각화_시연_한계.md').write_text(doc,encoding='utf-8')
print(json.dumps(stats,ensure_ascii=False,indent=2))
print('Saved six PNG/SVG figures, evidence tables and report.')
