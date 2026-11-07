"""Orthographic engineering drawings (third-angle projection) of case v2 from the assembly STL files.
Visible edges = sharp feature edges (>30 deg) + view silhouettes, with hidden-line removal by ray casting (embree)."""
import numpy as np, trimesh, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

CASE = '/home/claude/case/'
REN = '/tmp/claude-0/-home-claude/78aa3206-2c91-5dc8-8f0c-09913a22e00d/scratchpad/render/'
OUT = '/home/claude/final/fig/'

def load(names):
    ms = [trimesh.load(CASE + n) for n in names]
    m = trimesh.util.concatenate(ms)
    m.apply_translation(-m.bounds[0])
    return m

def view_edges(m, d, axes):
    """Return list of visible 2-D polylines for view direction d (unit vector pointing from the viewer into the scene)."""
    d = np.asarray(d, float)
    n = m.face_normals
    adj = m.face_adjacency; ang = m.face_adjacency_angles; e = m.face_adjacency_edges
    s1 = n[adj[:, 0]] @ d; s2 = n[adj[:, 1]] @ d
    keep = (ang > np.radians(30)) | (np.sign(s1) != np.sign(s2))
    # an edge whose both faces point away from the viewer can never be visible
    keep &= ~((s1 > 1e-6) & (s2 > 1e-6))
    E = e[keep]; V = m.vertices
    inter = trimesh.ray.ray_pyembree.RayMeshIntersector(m)
    lines = []
    pts_all, idx_all = [], []
    for k, (a, b) in enumerate(E):
        pa, pb = V[a], V[b]; L = np.linalg.norm(pb - pa)
        ns = max(2, int(L / 0.4) + 1)
        t = np.linspace(0, 1, ns)[:, None]
        pts_all.append(pa + (pb - pa) * t); idx_all.append(np.full(ns, k))
    P = np.vstack(pts_all); I = np.concatenate(idx_all)
    # nudge sample slightly towards the viewer so it does not hit its own faces first
    origins = P - d * 500.0
    locs, ray_idx, _ = inter.intersects_location(origins, np.tile(d, (len(P), 1)), multiple_hits=False)
    first = np.full(len(P), np.inf)
    first[ray_idx] = np.linalg.norm(locs - origins[ray_idx], axis=1)
    vis = first >= 500.0 - 0.08
    for k in range(len(E)):
        sel = np.where(I == k)[0]; v = vis[sel]; p2 = P[sel][:, axes]
        start = None
        for j in range(len(sel)):
            if v[j] and start is None: start = j
            if (not v[j] or j == len(sel) - 1) and start is not None:
                end = j if v[j] else j - 1
                if end > start: lines.append(p2[start:end + 1])
                start = None
    return lines

def dim_h(ax, x1, x2, y, text, off=8, fs=7):
    ax.annotate('', xy=(x1, y - off), xytext=(x2, y - off), arrowprops=dict(arrowstyle='<|-|>', lw=0.5, color='k', mutation_scale=6))
    for x in (x1, x2): ax.plot([x, x], [y - 1.5, y - off - 1.5], lw=0.35, color='k')
    ax.text((x1 + x2) / 2, y - off + 1.0, text, ha='center', va='bottom', fontsize=fs)

def dim_v(ax, y1, y2, x, text, off=8, fs=7):
    ax.annotate('', xy=(x + off, y1), xytext=(x + off, y2), arrowprops=dict(arrowstyle='<|-|>', lw=0.5, color='k', mutation_scale=6))
    for y in (y1, y2): ax.plot([x + 1.5, x + off + 1.5], [y, y], lw=0.35, color='k')
    ax.text(x + off + 1.2, (y1 + y2) / 2, text, ha='left', va='center', fontsize=fs, rotation=90)

def projection_symbol(ax, x, y):
    # third-angle projection symbol: cone frustum (right) and circles (left)
    ax.add_patch(plt.Circle((x, y), 3.2, fill=False, lw=0.5)); ax.add_patch(plt.Circle((x, y), 1.6, fill=False, lw=0.5))
    ax.plot([x + 6, x + 6, x + 13, x + 13, x + 6], [y - 1.6, y + 1.6, y + 3.2, y - 3.2, y - 1.6], lw=0.5, color='k')
    ax.plot([x - 4.5, x + 14.5], [y, y], lw=0.3, color='k', ls='-.')

def sheet(m, title, part, number, scale, render_png, notes, dims, fname, qty='1', mass=None):
    W, H = 420, 297
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(Rectangle((10, 10), W - 20, H - 20, fill=False, lw=1.0))
    ext = m.bounds[1] - m.bounds[0]; sx, sy, sz = ext * scale
    # third-angle layout: front view lower-left, top view above it, right view to its right
    fx, fy = 45, 80
    tx, ty = fx, fy + sz + 32
    rx, ry = fx + sx + 32, fy
    views = {'front': ((0, 1, 0), (0, 2), (fx, fy), False),
             'top': ((0, 0, -1), (0, 1), (tx, ty), False),
             'right': ((-1, 0, 0), (1, 2), (rx, ry), True)}
    for name, (d, axes, (ox, oy), flip) in views.items():
        for ln in view_edges(m, d, list(axes)):
            xy = ln * scale
            if flip: xy = np.c_[ext[1] * scale - xy[:, 0], xy[:, 1]]
            ax.plot(ox + xy[:, 0], oy + xy[:, 1], color='k', lw=0.45, solid_capstyle='round')
    lab = dict(fontsize=8, fontweight='bold', ha='center')
    ax.text(fx + sx / 2, fy - 22, 'FRONT VIEW', **lab); ax.text(tx + sx / 2, ty + sy + 6, 'TOP VIEW', **lab)
    ax.text(rx + sy / 2, ry - 22, 'RIGHT VIEW', **lab)
    # overall dimensions
    dim_h(ax, fx, fx + sx, fy, f'{ext[0]:.1f}')
    dim_v(ax, fy, fy + sz, rx + sy, f'{ext[2]:.1f}')
    dim_h(ax, rx, rx + sy, ry, f'{ext[1]:.1f}')
    for f in dims: f(ax, dict(fx=fx, fy=fy, tx=tx, ty=ty, rx=rx, ry=ry, s=scale, ext=ext))
    # isometric render
    im = Image.open(render_png); iw, ih = im.size; bw = 110; bh = bw * ih / iw
    if bh > 95: bh = 95; bw = bh * iw / ih
    ax.imshow(im, extent=(W - 20 - bw, W - 20, H - 25 - bh, H - 25), zorder=2)
    ax.text(W - 20 - bw / 2, H - 22 - bh - 7, 'ISOMETRIC VIEW (NOT TO SCALE)', **lab)
    # notes
    ny = 125
    ax.text(rx + sy + 40 if rx + sy + 40 < W - 150 else W - 150, ny, 'NOTES', fontsize=8, fontweight='bold')
    for i, t in enumerate(notes):
        ax.text(rx + sy + 40 if rx + sy + 40 < W - 150 else W - 150, ny - 6 - i * 5.2, f'{i + 1}. {t}', fontsize=6.6, va='top')
    # title block
    bx, by, bw2, bh2 = W - 10 - 180, 10, 180, 42
    ax.add_patch(Rectangle((bx, by), bw2, bh2, fill=False, lw=0.8))
    rows = [('TITLE', title), ('PART', part), ('MATERIAL', 'PLA (FDM 3D print)'), ('UNITS', 'mm'),
            ('SCALE', f'{scale:g}:1 on A3'), ('QTY', qty)]
    for i, (k, v) in enumerate(rows[:3]):
        y = by + bh2 - 7 - i * 8
        ax.text(bx + 3, y, k, fontsize=5.5, color='#555'); ax.text(bx + 25, y, v, fontsize=8 if i else 9, fontweight='bold' if i == 0 else None)
    for i, (k, v) in enumerate(rows[3:]):
        x = bx + 3 + i * 36; ax.text(x, by + 13, k, fontsize=5.5, color='#555'); ax.text(x, by + 7, v, fontsize=7.5)
    ax.text(bx + 112, by + 13, 'DWG NO.', fontsize=5.5, color='#555'); ax.text(bx + 112, by + 7, number, fontsize=8, fontweight='bold')
    ax.text(bx + 150, by + 13, 'SHEET', fontsize=5.5, color='#555'); ax.text(bx + 150, by + 7, '1 of 1', fontsize=7.5)
    ax.plot([bx, bx + bw2], [by + 18, by + 18], lw=0.5, color='k')
    ax.text(bx + 3, by + 2.5, 'G24 – Swinburne University of Technology · ENG40008 · Nov 2026 · Third-angle projection', fontsize=5.5)
    projection_symbol(ax, bx - 30, by + 21)
    ax.text(bx - 35, by + 12, 'THIRD ANGLE', fontsize=5.5)
    fig.savefig(OUT + fname, dpi=200, facecolor='white'); plt.close(fig)
    print('saved', fname, ext.round(1))

if __name__ == '__main__':
    base = load(['v2_assembly_base.stl']); lid = load(['v2_assembly_lid.stl'])
    pod = load(['v2_assembly_pod.stl']); cap = load(['v2_assembly_cap.stl'])
    asm = load(['v2_assembly_base.stl', 'v2_assembly_lid.stl', 'v2_assembly_pod.stl', 'v2_assembly_cap.stl'])
    std = ['General tolerance ±0.3 mm unless stated.', 'Print at 0.2 mm layer height, 3 walls, 20 % infill.', 'Remove sharp edges and stringing after printing.']
    sheet(asm, 'Raspberry Pi 5 case v2 – general assembly', 'Assembly (base, lid, camera pod, pod cap)', 'G24-CASE-00', 1.0,
          REN + 'c_case_front.png',
          ['Assembly shown with the camera pod upright.', 'Pi 5 fixed to base with 4 × M2.5 × 6 mm screws.',
           'Pod hinge: 1 × M3 × 20 mm bolt and nut through lid lugs.', 'Camera Module 3 held in pod rails by the pod cap.',
           'Parts list: see Table 8.1.'], [], 'DWG_00_assembly.png', qty='1 set')
    sheet(base, 'Raspberry Pi 5 case v2', 'Base', 'G24-CASE-01', 1.0, REN + 'c_p_base.png',
          ['Wall thickness 2.2 mm; outer corner radius 7 mm.', 'Inner space 87 × 59 × 28 mm for the Pi 5 (85 × 56 mm).',
           'Port openings: USB-C, 2 × micro-HDMI, Ethernet, 2 × USB, microSD.', 'Side vent slots on the long wall.'] + std, [], 'DWG_01_base.png')
    sheet(lid, 'Raspberry Pi 5 case v2', 'Vented lid', 'G24-CASE-02', 1.0, REN + 'c_p_lid.png',
          ['Hexagonal vent field above the processor (hex 5.2 mm).', 'GPIO/cable slot and camera ribbon slot.',
           'Two hinge lugs for the camera pod (M3 bolt).'] + std, [], 'DWG_02_lid.png')
    sheet(pod, 'Raspberry Pi 5 case v2', 'Camera pod', 'G24-CASE-03', 2.0, REN + 'c_p_pod.png',
          ['Wall thickness 2.0 mm; cavity height 25 mm.', 'Side rails hold the 25 × 24 mm Camera Module 3 board.',
           'Lens window in the front wall; ribbon exit in the floor.', 'Hinge boss under the front for an M3 bolt.'] + std, [], 'DWG_03_pod.png')
    sheet(cap, 'Raspberry Pi 5 case v2', 'Pod cap', 'G24-CASE-04', 2.0, REN + 'c_p_cap.png',
          ['Push-fit into the top of the camera pod.', 'Keeps the camera board in its rails.'] + std, [], 'DWG_04_cap.png')
