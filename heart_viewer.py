#!/usr/bin/env python3
"""
heart_viewer.py - interactive 3D heart viewer for Raspberry Pi 5
FYP-2: CT + MRI -> 3D Slicer -> STL/OBJ -> Pi

Usage:
    python3 heart_viewer.py export/ct export/mr     (folders of STL files)
    python3 heart_viewer.py heart.obj               (a single OBJ also works)

Each folder/file you give is one LAYER (e.g. CT and MR).

Controls:
    Left-drag     rotate
    Right-drag    pan
    Scroll        zoom
    1-9           show / hide a structure (in every layer)
    F1-F4         show / hide layer 1-4      (C = CT layer, M = MR layer)
    T             see-through myocardium (shows the chambers inside)
    A             auto-rotate on / off
    R             reset view
    S             save screenshot
    Q / Esc       quit
"""
import glob
import os
import struct
import sys
import time

# On Wayland desktops (Ubuntu, Raspberry Pi OS Bookworm) PyOpenGL picks EGL,
# but the pygame window is X11 (XWayland) -> "no valid context". Force GLX.
# Same problem over SSH and on Pi OS, so use GLX whenever an X display exists.
if "PYOPENGL_PLATFORM" not in os.environ and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    os.environ["PYOPENGL_PLATFORM"] = "glx"
if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    sys.exit("No screen found. Run this in a terminal on the Pi desktop (e.g. inside VNC), not a plain SSH session.")

import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE
from OpenGL.GL import *   # noqa: F401,F403

# Colours for the MM-WHS structures (matched by name, case-insensitive)
COLOURS = {
    "lv myocardium":    (0.85, 0.55, 0.40),
    "left atrium":      (0.95, 0.35, 0.35),
    "lv blood cavity":  (0.80, 0.10, 0.15),
    "right atrium":     (0.35, 0.55, 0.95),
    "rv blood cavity":  (0.15, 0.30, 0.85),
    "ascending aorta":  (0.95, 0.75, 0.20),
    "pulmonary artery": (0.30, 0.75, 0.85),
}
FALLBACK = [(0.8, 0.3, 0.3), (0.3, 0.5, 0.9), (0.9, 0.7, 0.2), (0.4, 0.8, 0.5),
            (0.7, 0.4, 0.8), (0.9, 0.5, 0.2), (0.3, 0.8, 0.8), (0.8, 0.8, 0.8)]
SEE_THROUGH_WORDS = ("myocard", "heart")     # what T makes transparent
SECOND_LAYER_ALPHA = 0.35                    # MR drawn see-through over CT


# ----------------------------------------------------------------- loading
def read_stl(path):
    """Return (n,3,3) triangle array from a binary or ASCII STL."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + n * 50 == len(data):                        # binary
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                                      ("a", "<u2")]), count=n, offset=84)
            return rec["v"].astype(np.float32)
    pts = [list(map(float, l.split()[1:4])) for l in data.decode(errors="ignore").splitlines()
           if l.strip().startswith("vertex")]
    return np.array(pts, dtype=np.float32).reshape(-1, 3, 3)


def read_obj(path):
    """Return list of (name, (n,3,3) triangles, colour) from an OBJ (+ .mtl)."""
    folder = os.path.dirname(os.path.abspath(path))
    verts, mats, groups = [], {}, []

    def mtl_load(p):
        cur = None
        if os.path.exists(p):
            for line in open(p):
                s = line.split()
                if s and s[0] == "newmtl":
                    cur = " ".join(s[1:]); mats[cur] = None
                elif s and s[0] == "Kd" and cur:
                    mats[cur] = tuple(float(x) for x in s[1:4])

    for line in open(path):
        s = line.split()
        if not s:
            continue
        if s[0] == "v":
            verts.append([float(s[1]), float(s[2]), float(s[3])])
        elif s[0] == "mtllib":
            p = os.path.join(folder, line.strip()[7:].strip())
            mtl_load(p if os.path.exists(p) else os.path.splitext(path)[0] + ".mtl")
        elif s[0] in ("g", "o") or (s[0] == "usemtl" and (not groups or groups[-1][1])):
            groups.append([" ".join(s[1:]), [], None])
            if s[0] == "usemtl":
                groups[-1][2] = mats.get(" ".join(s[1:]))
        elif s[0] == "usemtl":
            groups[-1][2] = mats.get(" ".join(s[1:]))
        elif s[0] == "f":
            if not groups:
                groups.append(["model", [], None])
            idx = [int(v.split("/")[0]) for v in s[1:]]
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
            for k in range(1, len(idx) - 1):
                groups[-1][1].append((idx[0], idx[k], idx[k + 1]))
    V = np.array(verts, dtype=np.float32)
    return [(g[0], V[np.array(g[1])], g[2]) for g in groups if g[1]]


def clean_name(stem):
    """'CT heart_LV myocardium' -> 'LV myocardium'."""
    low = stem.lower()
    for key in COLOURS:
        if key in low:
            return stem[low.index(key):low.index(key) + len(key)]
    return stem.split("_", 1)[-1] if "_" in stem else stem


def smooth_normals(tri):
    """Per-vertex normals averaged over shared corners -> smooth shading."""
    flat = tri.reshape(-1, 3)
    _, inv = np.unique(np.round(flat, 3), axis=0, return_inverse=True)
    inv = inv.ravel()
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    acc = np.zeros((inv.max() + 1, 3), dtype=np.float64)
    np.add.at(acc, inv, np.repeat(fn, 3, axis=0))
    acc /= np.linalg.norm(acc, axis=1, keepdims=True) + 1e-12
    return acc[inv].astype(np.float32)


def load_layers(paths):
    layers = []
    for li, path in enumerate(paths):
        items = []
        if os.path.isdir(path):
            for f in sorted(glob.glob(os.path.join(path, "*.stl"))):
                items.append((clean_name(os.path.splitext(os.path.basename(f))[0]), read_stl(f), None))
        elif path.lower().endswith(".stl"):
            items.append((clean_name(os.path.splitext(os.path.basename(path))[0]), read_stl(path), None))
        else:
            items = read_obj(path)
        if not items:
            sys.exit("Nothing to load in " + path)
        name = os.path.basename(os.path.normpath(path)).upper()
        parts = []
        for i, (pname, tri, col) in enumerate(items):
            col = COLOURS.get(pname.lower(), col or FALLBACK[i % len(FALLBACK)])
            parts.append({"name": pname, "tri": tri, "color": col})
        layers.append({"name": name, "parts": parts, "visible": True,
                       "alpha": 1.0 if li == 0 else SECOND_LAYER_ALPHA})

    # one shared centre/scale so CT and MR stay aligned
    allv = np.concatenate([p["tri"].reshape(-1, 3) for L in layers for p in L["parts"]])
    centre = (allv.min(0) + allv.max(0)) / 2
    scale = 2.0 / float(np.max(allv.max(0) - allv.min(0)))
    for L in layers:
        for p in L["parts"]:
            tri = (p["tri"] - centre) * scale
            p["v"] = np.ascontiguousarray(tri.reshape(-1, 3), dtype=np.float32)
            p["n"] = np.ascontiguousarray(smooth_normals(tri))
            p["count"] = len(p["v"])
            del p["tri"]
    return layers


# --------------------------------------------------------------- rendering
def upload(layers):
    for L in layers:
        for p in L["parts"]:
            p["vbo_v"], p["vbo_n"] = glGenBuffers(2)
            glBindBuffer(GL_ARRAY_BUFFER, p["vbo_v"])
            glBufferData(GL_ARRAY_BUFFER, p["v"].nbytes, p["v"], GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, p["vbo_n"])
            glBufferData(GL_ARRAY_BUFFER, p["n"].nbytes, p["n"], GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)


def draw(p, alpha):
    glColor4f(*p["color"], alpha)
    glBindBuffer(GL_ARRAY_BUFFER, p["vbo_v"])
    glVertexPointer(3, GL_FLOAT, 0, None)
    glBindBuffer(GL_ARRAY_BUFFER, p["vbo_n"])
    glNormalPointer(GL_FLOAT, 0, None)
    glDrawArrays(GL_TRIANGLES, 0, p["count"])


def setup_gl(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    top = 0.05 * np.tan(np.radians(40) / 2)          # 40 deg field of view
    right = top * w / max(h, 1)
    glFrustum(-right, right, -top, top, 0.05, 50)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE)
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.25, 0.25, 0.25, 1))
    glLightfv(GL_LIGHT0, GL_POSITION, (0.5, 1.0, 1.0, 0.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.9, 0.9, 0.9, 1))
    glLightfv(GL_LIGHT1, GL_POSITION, (-1.0, -0.5, 0.5, 0.0))
    glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.35, 0.35, 0.4, 1))
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.3, 0.3, 0.3, 1))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 30)
    glEnable(GL_NORMALIZE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.08, 0.09, 0.12, 1)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)


# Slicer exports in LPS (x=left, y=posterior, z=superior):
# turn it so the patient faces the viewer, head up.
LPS_TO_VIEW = np.array([[1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]], dtype=np.float32)


def flag(name, default=None, cast=str):
    """--name VALUE from the command line (test options)."""
    if name in sys.argv:
        i = sys.argv.index(name)
        return cast(sys.argv[i + 1]) if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--") else True
    return default


def save_log(path, rows, header):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join("" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in r) + "\n")
    print("Log saved:", path, f"({len(rows)} frames)")


def main():
    # test options: --bench SECONDS (auto-rotate, then quit), --log file.csv, --uncapped (no 60 FPS cap)
    bench = flag("--bench", None, float)
    log_path = flag("--log", None)
    uncapped = flag("--uncapped", False) is True
    skip = {sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a in ("--bench", "--log")}
    paths = [a for a in sys.argv[1:] if not a.startswith("--") and a not in skip] or ["export/ct", "export/mr"]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        sys.exit("Give a folder of STL files or an OBJ file, e.g.  python3 heart_viewer.py export/ct export/mr")
    t0 = time.time()
    layers = load_layers(paths)
    structures = []                                   # unique names, in order
    for L in layers:
        for p in L["parts"]:
            if p["name"] not in structures:
                structures.append(p["name"])
    hidden = set()
    print(f"Loaded in {time.time() - t0:.1f}s")
    for li, L in enumerate(layers):
        tris = sum(p["count"] for p in L["parts"]) // 3
        print(f"  Layer F{li + 1}: {L['name']:<6} {len(L['parts'])} structures, {tris} triangles")
    for i, s in enumerate(structures):
        print(f"  [{i + 1}] {s}")
    print(__doc__.split("Controls:")[1])

    pygame.init()
    W, H = 1280, 720
    pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL | RESIZABLE)
    print("OpenGL:", glGetString(GL_VERSION).decode(), "|", glGetString(GL_RENDERER).decode())
    setup_gl(W, H)
    upload(layers)

    rot_x, rot_y, dist, pan = 0.0, 0.0, 4.0, [0.0, 0.0]
    auto, see_through = False, False
    clock = pygame.time.Clock()
    frames, fps_t, shot = 0, time.time(), 0
    if bench:
        auto = True
    log_rows, t_start, t_prev = [], time.time(), time.time()
    tris_total = sum(p["count"] for L in layers for p in L["parts"]) // 3

    def finish():
        if len(log_rows) > 30:
            dt = np.array([r[1] for r in log_rows[10:]]) / 1000.0        # skip warm-up frames
            f = 1.0 / dt
            print(f"RESULT viewer | layers: {'+'.join(L['name'] for L in layers)} | {tris_total} triangles | "
                  f"{'uncapped' if uncapped else 'capped 60'} | frames {len(dt)} | "
                  f"mean {len(dt) / dt.sum():.1f} FPS | SD {f.std():.1f} | min {f.min():.1f} | 5th pct {np.percentile(f, 5):.1f}")
            if log_path:
                save_log(log_path, log_rows, ["t_s", "frame_ms", "fps_inst", "triangles"])

    def toggle_layer(i):
        if i < len(layers):
            layers[i]["visible"] = not layers[i]["visible"]
            print(layers[i]["name"], "layer", "ON" if layers[i]["visible"] else "OFF")

    while True:
        if bench and time.time() - t_start > bench:
            finish()
            return
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                finish()
                return
            if e.type == pygame.VIDEORESIZE:
                W, H = e.w, e.h
                pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL | RESIZABLE)
                setup_gl(W, H)
            if e.type == pygame.MOUSEWHEEL:
                dist = float(np.clip(dist * (0.9 if e.y > 0 else 1.1), 1.0, 20))
            if e.type == pygame.MOUSEMOTION:
                if e.buttons[0]:
                    rot_y += e.rel[0] * 0.5
                    rot_x += e.rel[1] * 0.5
                elif e.buttons[2]:
                    pan[0] += e.rel[0] * 0.003 * dist
                    pan[1] -= e.rel[1] * 0.003 * dist
            if e.type == pygame.KEYDOWN:
                k = e.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    finish()
                    return
                if pygame.K_1 <= k <= pygame.K_9:
                    i = k - pygame.K_1
                    if i < len(structures):
                        hidden.symmetric_difference_update({structures[i]})
                        print(structures[i], "OFF" if structures[i] in hidden else "ON")
                elif pygame.K_F1 <= k <= pygame.K_F4:
                    toggle_layer(k - pygame.K_F1)
                elif k in (pygame.K_c, pygame.K_m):
                    want = "CT" if k == pygame.K_c else "MR"
                    for i, L in enumerate(layers):
                        if want in L["name"]:
                            toggle_layer(i)
                elif k == pygame.K_t:
                    see_through = not see_through
                elif k == pygame.K_a:
                    auto = not auto
                elif k == pygame.K_r:
                    rot_x, rot_y, dist, pan = 0.0, 0.0, 4.0, [0.0, 0.0]
                elif k == pygame.K_s:
                    shot += 1
                    data = glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE)
                    img = pygame.image.fromstring(data, (W, H), "RGB", True)
                    name = f"heart_screenshot_{shot}.png"
                    pygame.image.save(img, name)
                    print("Saved", name)

        if auto:
            rot_y += 0.6

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(pan[0], pan[1], -dist)
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_y, 0, 1, 0)
        glMultMatrixf(LPS_TO_VIEW)

        todo = []
        n_visible = sum(L["visible"] for L in layers)
        for L in layers:
            if not L["visible"]:
                continue
            for p in L["parts"]:
                if p["name"] in hidden:
                    continue
                a = 1.0 if n_visible == 1 else L["alpha"]     # a layer on its own is drawn solid
                if see_through and any(w in p["name"].lower() for w in SEE_THROUGH_WORDS):
                    a = min(a, 0.25)
                todo.append((p, a))
        for p, a in todo:                       # solid first
            if a >= 0.99:
                draw(p, a)
        glDepthMask(GL_FALSE)                   # then see-through
        for p, a in todo:
            if a < 0.99:
                draw(p, a)
        glDepthMask(GL_TRUE)

        pygame.display.flip()
        if not uncapped:
            clock.tick(60)
        now = time.time()
        log_rows.append([now - t_start, (now - t_prev) * 1000.0, 1.0 / max(now - t_prev, 1e-6), tris_total])
        t_prev = now

        frames += 1
        if time.time() - fps_t >= 1.0:
            pygame.display.set_caption(f"Heart viewer  |  {frames / (time.time() - fps_t):.1f} FPS")
            frames, fps_t = 0, time.time()


if __name__ == "__main__":
    main()
