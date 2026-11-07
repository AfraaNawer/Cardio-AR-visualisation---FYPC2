import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
T='/home/claude/results/T2/'
# ---- Fig: T2 bar chart
s=pd.read_csv('/home/claude/results/T2_summary.csv')
names={'ct':'CT viewer\n(690,924 tri)','mr':'MRI viewer\n(110,042 tri)','both':'CT + MRI viewer\n(800,966 tri)','ar':'AR overlay\n(CT + MRI + pose)'}
order=['ct','mr','both','ar']; cols=['#9ecae1','#4292c6','#08519c']
fig,ax=plt.subplots(figsize=(8,4.4),dpi=200)
x=np.arange(4); w=0.26
for r in [1,2,3]:
    v=[s[(s['mode']==m)&(s.run==r)].fps.iloc[0] for m in order]
    b=ax.bar(x+(r-2)*w,v,w,color=cols[r-1],label=f'Run {r}')
    for xi,vi in zip(x+(r-2)*w,v): ax.text(xi,vi*1.06 if vi>30 else vi*0.62,f'{vi:.0f}' if vi>30 else f'{vi:.1f}',ha='center',fontsize=7.5,color='black' if vi>30 or r==1 else 'white')
ax.axhline(20,color='#c0392b',ls='--',lw=1.3); ax.text(3.55,21.5,'Target 20 FPS',color='#c0392b',fontsize=8.5,ha='right')
ax.set_yscale('log'); ax.set_ylim(8,500); ax.set_yticks([10,20,50,100,200,500]); ax.set_yticklabels(['10','20','50','100','200','500'])
ax.set_xticks(x); ax.set_xticklabels([names[m] for m in order]); ax.set_ylabel('Mean frame rate (FPS, log scale)')
ax.legend(frameon=False,ncol=3,loc='upper right'); ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('fig/T2_fps_bar.png'); plt.close()
# ---- Fig: frame time trace (run 3)
fig,axs=plt.subplots(2,1,figsize=(8,4.6),dpi=200,sharex=True)
for m,c,l in [('ct','#4292c6','CT'),('both','#08519c','CT + MRI'),('mr','#9ecae1','MRI')]:
    d=pd.read_csv(T+f'T2_{m}_3.csv').iloc[10:]; fps=(1000/d.frame_ms).rolling(30,center=True).mean()
    axs[0].plot(d.t_s,fps,color=c,lw=1,label=l)
axs[0].set_yscale('log'); axs[0].set_yticks([20,50,100,200,300]); axs[0].set_yticklabels(['20','50','100','200','300']); axs[0].minorticks_off(); axs[0].axhline(20,color='#c0392b',ls='--',lw=1); axs[0].set_ylabel('FPS (log)'); axs[0].legend(frameon=False,ncol=3,fontsize=8,loc='lower right'); axs[0].set_title('3D viewer (run 3, 30-frame moving average)',fontsize=9,loc='left')
d=pd.read_csv(T+'T2_ar_3.csv'); d['t']=d.t_s-d.t_s.iloc[0]
axs[1].plot(d.t,(1000/d.frame_ms).rolling(15,center=True).mean(),color='#e6550d',lw=1,label='AR frame rate')
nb=d.body_detected==0; axs[1].fill_between(d.t,0,40,where=nb,color='#bbb',alpha=.5,step='mid',label='no body in view')
axs[1].axhline(20,color='#c0392b',ls='--',lw=1); axs[1].set_ylim(0,30); axs[1].set_ylabel('FPS'); axs[1].set_xlabel('Time (s)')
axs[1].legend(frameon=False,ncol=2,fontsize=8,loc='lower right'); axs[1].set_title('AR overlay (run 3, 15-frame moving average)',fontsize=9,loc='left')
for a in axs: a.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('fig/T2_trace.png'); plt.close()
