import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np
from PIL import Image
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
R='/home/claude/results/'; S='/tmp/claude-0/-home-claude/78aa3206-2c91-5dc8-8f0c-09913a22e00d/scratchpad/'
# T3
a=pd.read_csv(R+'T3/T3_ar_readings.csv'); c=pd.read_csv(R+'T3/T3_camera_only_readings.csv')
la=(a.real_s-a.ar_s)*1000; lc=(c.real_s-c.shown_s)*1000
fig,ax=plt.subplots(figsize=(7,3.8),dpi=200)
rng=np.random.default_rng(1)
for i,(v,col,lab) in enumerate([(lc,'#9ecae1','Camera preview only\n(rpicam-hello → VNC)'),(la,'#e6550d','Full AR pipeline\n(ar_body.py → VNC)')]):
    ax.scatter(i+rng.uniform(-.12,.12,len(v)),v,color=col,s=28,zorder=3,edgecolor='#333',lw=.4)
    ax.errorbar(i+.28,v.mean(),yerr=v.std(),fmt='o',color='black',capsize=5,zorder=4)
    ax.text(i+.36,v.mean(),f'{v.mean():.0f} ± {v.std():.0f} ms',va='center',fontsize=9)
ax.axhline(100,color='#c0392b',ls='--',lw=1.2); ax.text(1.75,103,'Target 100 ms',color='#c0392b',fontsize=8.5,ha='right')
ax.annotate('',xy=(1.0,la.mean()),xytext=(1.0,lc.mean()),arrowprops=dict(arrowstyle='<->',color='#555'))
ax.text(0.62,(la.mean()+lc.mean())/2,f'+{la.mean()-lc.mean():.0f} ms\nAR processing',fontsize=8.5,color='#555',va='center')
ax.set_xticks([0,1]); ax.set_xticklabels(['Camera preview only\n(rpicam-hello → VNC)','Full AR pipeline\n(ar_body.py → VNC)'])
ax.set_xlim(-.5,1.85); ax.set_ylim(0,280); ax.set_ylabel('Camera-to-display latency (ms)')
ax.spines[['top','right']].set_visible(False); plt.tight_layout(); plt.savefig('fig/T3_latency.png'); plt.close()
# T4
t=pd.read_csv(R+'T4/T4_summary.csv',comment='#')
fig,axs=plt.subplots(1,2,figsize=(9,3.8),dpi=200)
ax=axs[0]; ax.plot([0,2.3],[0,2.3],color='#999',ls='--',lw=1,label='Ideal (estimate = true)')
ax.errorbar(t.true_m,t.est_median_m,yerr=[t.est_median_m-t.q10,t.q90-t.est_median_m],fmt='o',color='#08519c',capsize=4,label='Median estimate (10–90 % range)')
for x,y in zip(t.true_m,t.est_median_m): ax.text(x+.05,y-.13,f'{y-x:+.2f} m',fontsize=8.5)
ax.set_xlim(0,2.3); ax.set_ylim(0,2.5); ax.set_xlabel('Measured distance (m)'); ax.set_ylabel('Estimated distance (m)'); ax.legend(frameon=False,fontsize=8,loc='upper left'); ax.set_title('(a) Distance estimate',fontsize=10,loc='left')
ax=axs[1]; x=np.arange(4); w=.36
ax.bar(x-w/2,t.jitter_x_mm,w,color='#4292c6',label='Horizontal'); ax.bar(x+w/2,t.jitter_y_mm,w,color='#08519c',label='Vertical')
ax.set_xticks(x); ax.set_xticklabels([f'{d:.1f} m' for d in t.true_m]); ax.set_ylabel('Heart-centre SD (mm)'); ax.legend(frameon=False,fontsize=8); ax.set_title('(b) Anchor jitter while standing still',fontsize=10,loc='left')
for a_ in axs: a_.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('fig/T4_distance_jitter.png'); plt.close()
# T5 two person
d=pd.read_csv(R+'T5/T5_two_person.csv')
fig,ax=plt.subplots(figsize=(7,3.4),dpi=200); x=np.arange(3); w=.36
for k,(run,col,lab) in enumerate([('poses1','#4292c6','Single-person mode (num_poses = 1)'),('poses2','#e6550d','Two-person mode (num_poses = 2)')]):
    v=d[d.run==run].heart_jumps.values; ax.bar(x+(k-.5)*w,v,w,color=col,label=lab)
    for xi,vi in zip(x+(k-.5)*w,v): ax.text(xi,vi+.5,str(vi),ha='center',fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(['Subject A alone\n(0–15 s, includes walking in)','A and B in view\n(15–30 s)','B alone\n(30–45 s)'])
ax.set_ylabel('Heart jumps (> 0.5 shoulder width)'); ax.legend(frameon=False,fontsize=8); ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('fig/T5_two_person.png'); plt.close()
# photos
Image.open(S+'vid/ar_mid.png').crop((215,120,1068,578)).save('fig/ar_overlay_photo.png')
Image.open(S+'t3/f15.png').crop((0,120,1080,720)).save('fig/setup_photo.png')
Image.open(R+'T3/T3_example.png').crop((66,40,1913,960)).save('fig/T3_example.png')
print('ok')
