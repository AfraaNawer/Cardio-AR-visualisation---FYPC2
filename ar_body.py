#!/usr/bin/env python3
"""
ar_body.py - markerless AR: shows the heart on a person's chest (no marker)
FYP-2  |  Raspberry Pi 5 + Camera Module 3 + MediaPipe Pose

The camera finds the person's shoulders and hips (MediaPipe Pose Landmarker),
works out where the heart sits in the chest (slightly to the person's LEFT of
the midline, about a third of the way from shoulders to hips), estimates the
distance from the shoulder width, and draws the 3D heart there at real size.

Needs (same folder): heart_viewer.py, ar_heart.py, export/, pose_landmarker_lite.task

Usage (terminal on the Pi desktop / VNC):
    ~/mpenv/bin/python ar_body.py export/ct export/mr

Test options (all optional, normal use is unchanged):
    --log FILE.csv      log every frame (FPS, tracking, heart position, marker error)
    --bench SECONDS     stop automatically after this many seconds
    --poses N           let MediaPipe look for up to N people (default 1)
    --marker-mm MM      placement test: a printed ArUco marker (side MM mm) taped to the
                        chest where the heart should be; logs heart-to-marker error in mm

Keys:
    1-9      show / hide a structure          C / M   CT / MRI layer
    T        see-through myocardium           G       ghost mode (whole heart see-through)
    + / -    bigger / smaller heart           Left / Right arrows  turn the heart
    Up / Down arrows  tilt the heart forwards / backwards
    K        show body key points (debug)     R       reset

Goggles mode (3.5" SPI LCD inside the 3D-printed viewer):
    ~/mpenv/bin/python ar_body.py export/ct export/mr --goggles
    The full AR view stays on the HDMI / VNC screen for the audience, and a
    side-by-side (left eye | right eye) copy is sent to the LCD for the wearer.
    S        save screenshot                  Q / Esc quit
"""
import os
import sys
import time

import cv2
import numpy as np

import heart_viewer as hv                       # sets the OpenGL/GLX fix
from heart_viewer import load_layers, upload, draw, setup_gl, LPS_TO_VIEW, SEE_THROUGH_WORDS, flag, save_log
import ar_heart as ah                           # camera, background and projection helpers
from ar_heart import W, H, K, PiCamera, make_background_texture, draw_background, camera_projection

import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *   # noqa: F401,F403

# ---- anatomy settings ---------------------------------------------------------
SHOULDER_M = 0.38     # typical adult shoulder width (m) -> used to estimate distance
HEART_M = 0.13        # size of the heart model's largest dimension (m)
HEART_DOWN = 0.30     # heart centre: fraction of the way from shoulders to hips
HEART_LEFT = 0.20     # shift toward the person's LEFT, as a fraction of shoulder width
GHOST_ALPHA = 0.6
SMOOTH = 0.6          # 0 = no smoothing, 0.9 = very smooth
HOLD_FRAMES = 8
POSE_INPUT = (640, 360)   # pose runs on a smaller image (faster)
MODEL = "pose_landmarker_lite.task"

L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24      # MediaPipe landmark numbers (person's left/right)


class PoseFinder:
    def __init__(self, num_poses=1):
        if not os.path.exists(MODEL):
            sys.exit(f"Missing {MODEL}. Download it first (see instructions).")
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        self.mp = mp
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=MODEL),
            running_mode=vision.RunningMode.VIDEO, num_poses=num_poses,
            min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.lm = vision.PoseLandmarker.create_from_options(opts)
        self.t0 = time.time()
        self.last_ts = -1
        self.n_people = 0

    def find(self, frame_bgr):
        """Return dict of pixel points for shoulders/hips, or None."""
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, POSE_INPUT, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        ts = int((time.time() - self.t0) * 1000)
        ts = max(ts, self.last_ts + 1)
        self.last_ts = ts
        res = self.lm.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), ts)
        self.n_people = len(res.pose_landmarks)
        if not res.pose_landmarks:
            return None
        p = res.pose_landmarks[0]
        vis = lambda i: (getattr(p[i], "visibility", 1.0) or 0) > 0.5
        if not (vis(L_SH) and vis(R_SH)):
            return None
        px = lambda i: np.array([p[i].x * W, p[i].y * H])
        pts = {"ls": px(L_SH), "rs": px(R_SH)}
        if vis(L_HIP) and vis(R_HIP):
            pts["lh"], pts["rh"] = px(L_HIP), px(R_HIP)
        return pts


class FakePose:
    """For testing without a camera/model: a person standing in the middle."""
    n_people = 1

    def find(self, frame):
        return {"ls": np.array([780.0, 300.0]), "rs": np.array([500.0, 310.0]),
                "lh": np.array([740.0, 640.0]), "rh": np.array([540.0, 645.0])}


class FakeCamera:
    def read(self):
        img = np.full((H, W, 3), 110, np.uint8)
        cv2.ellipse(img, (640, 200), (60, 75), 0, 0, 360, (150, 170, 200), -1)              # head
        cv2.fillPoly(img, [np.int32([[500, 310], [780, 300], [740, 720], [540, 720]])], (80, 60, 50))  # torso
        return img

    def stop(self):
        pass


def heart_anchor(pts):
    """Heart centre in pixels, shoulder width in pixels, body roll angle (deg)."""
    ls, rs = pts["ls"], pts["rs"]
    across = ls - rs                              # points to the person's left (image right)
    sw = float(np.linalg.norm(across))
    mid_sh = (ls + rs) / 2
    if "lh" in pts:
        down = (pts["lh"] + pts["rh"]) / 2 - mid_sh
    else:                                         # no hips visible: assume torso ~1.3 x shoulder width
        perp = np.array([-across[1], across[0]]) / max(sw, 1e-6)
        down = perp * sw * 1.3 * np.sign(perp[1] or 1)
    centre = mid_sh + HEART_DOWN * down + HEART_LEFT * across
    roll = np.degrees(np.arctan2(across[1], across[0]))
    return centre, sw, roll


class GogglesOut:
    """Copies each rendered frame to the 3.5-inch SPI LCD as a side-by-side pair."""
    def __init__(self, fb=None):
        self.dev = fb or self.find_fb()
        base = os.path.basename(self.dev)
        w, h = open(f"/sys/class/graphics/{base}/virtual_size").read().strip().split(",")
        self.w, self.h = int(w), int(h)
        self.bpp = int(open(f"/sys/class/graphics/{base}/bits_per_pixel").read())
        self.f = open(self.dev, "wb")
        print(f"Goggles display: {self.dev} {self.w}x{self.h} {self.bpp}-bit")

    @staticmethod
    def find_fb():
        import glob
        for p in sorted(glob.glob("/sys/class/graphics/fb*")):
            name = open(p + "/name").read().strip().lower()
            if any(k in name for k in ("ili", "fb_", "st7", "hx8", "waveshare", "mipi")):
                return "/dev/" + os.path.basename(p)
        sys.exit("No SPI LCD framebuffer found. Install the LCD driver (dtoverlay) first, or pass --fb /dev/fbN")

    def show(self, rgb):
        """rgb: HxWx3 uint8 image of the full AR view."""
        eye_w = self.w // 2
        H, W = rgb.shape[:2]
        cw = int(H * eye_w / self.h)                      # crop centre to each eye's aspect
        x0 = max(0, (W - cw) // 2)
        eye = cv2.resize(rgb[:, x0:x0 + cw], (eye_w, self.h), interpolation=cv2.INTER_AREA)
        pair = np.concatenate([eye, eye], axis=1)         # same image for both eyes
        if self.bpp == 16:
            r = (pair[:, :, 0].astype(np.uint16) >> 3) << 11
            g = (pair[:, :, 1].astype(np.uint16) >> 2) << 5
            b = pair[:, :, 2].astype(np.uint16) >> 3
            data = (r | g | b).astype("<u2").tobytes()
        else:
            data = np.dstack([pair[:, :, 2], pair[:, :, 1], pair[:, :, 0], np.full(pair.shape[:2], 255, np.uint8)]).tobytes()
        self.f.seek(0); self.f.write(data); self.f.flush()


def find_marker(frame):
    """Centre (px) and mean side length (px) of the first ArUco marker in view, or None."""
    corners, ids, _ = ah.detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None or len(corners) == 0:
        return None
    c = corners[0].reshape(4, 2)
    side = float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))
    return c.mean(axis=0), side


def main():
    log_path = flag("--log", None)
    bench = flag("--bench", None, float)
    num_poses = int(flag("--poses", 1, int))
    marker_mm = flag("--marker-mm", None, float)
    skip = {sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a in ("--log", "--bench", "--poses", "--marker-mm", "--fb")}
    paths = [p for p in sys.argv[1:] if not p.startswith("--") and p not in skip] or ["export/ct", "export/mr"]
    fake = "--fake" in sys.argv
    goggles = None
    if "--goggles" in sys.argv:
        fb = sys.argv[sys.argv.index("--fb") + 1] if "--fb" in sys.argv else None
        goggles = GogglesOut(fb)
    layers = load_layers(paths)
    structures = []
    for L in layers:
        for p in L["parts"]:
            if p["name"] not in structures:
                structures.append(p["name"])
    print(__doc__.split("Keys:")[1])
    for i, s_ in enumerate(structures):
        print(f"  [{i + 1}] {s_}")

    cam = FakeCamera() if fake else PiCamera()
    finder = FakePose() if fake else PoseFinder(num_poses)

    pygame.init()
    pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("AR heart - body tracking")
    setup_gl(W, H)
    upload(layers)
    tex = make_background_texture()

    hidden, see_through, ghost, show_pts = set(), False, False, False
    scale, spin, tilt = 1.0, 0.0, 0.0
    state, lost = None, 99            # smoothed (cx, cy, sw, roll)
    fps, prev, shot, frame_times = 0.0, time.time(), 0, []
    log_rows, t_start = [], time.time()

    try:
        while True:
            if bench and time.time() - t_start > bench:
                return
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return
                if e.type == pygame.KEYDOWN:
                    k = e.key
                    if k in (pygame.K_q, pygame.K_ESCAPE):
                        return
                    if pygame.K_1 <= k <= pygame.K_9 and k - pygame.K_1 < len(structures):
                        hidden.symmetric_difference_update({structures[k - pygame.K_1]})
                    elif k in (pygame.K_c, pygame.K_m):
                        want = "CT" if k == pygame.K_c else "MR"
                        for L in layers:
                            if want in L["name"]:
                                L["visible"] = not L["visible"]
                    elif k == pygame.K_t:
                        see_through = not see_through
                    elif k == pygame.K_g:
                        ghost = not ghost
                    elif k == pygame.K_k:
                        show_pts = not show_pts
                    elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        scale *= 1.1
                    elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        scale /= 1.1
                    elif k == pygame.K_r:
                        scale, spin, tilt = 1.0, 0.0, 0.0
                    elif k == pygame.K_s:
                        shot += 1
                        data = glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE)
                        img = pygame.image.fromstring(data, (W, H), "RGB", True)
                        name = f"ar_body_{shot}.png"
                        pygame.image.save(img, name)
                        print("Saved", name)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                spin -= 2
            if keys[pygame.K_RIGHT]:
                spin += 2
            if keys[pygame.K_UP]:
                tilt -= 2
            if keys[pygame.K_DOWN]:
                tilt += 2

            t_frame = time.time()
            frame = cam.read()
            pts = finder.find(frame)
            raw = None
            mk = find_marker(frame) if marker_mm else None
            if pts:
                c, sw, roll = heart_anchor(pts)
                raw = (c, sw, roll)
                new = np.array([c[0], c[1], sw, roll])
                if state is not None and lost < HOLD_FRAMES:
                    new = SMOOTH * state + (1 - SMOOTH) * new
                state, lost = new, 0
                if show_pts:
                    for q in pts.values():
                        cv2.circle(frame, tuple(int(v) for v in q), 8, (0, 255, 0), -1)
                    cv2.drawMarker(frame, (int(state[0]), int(state[1])), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
                    a = tuple(int(v) for v in pts["ls"]); b = tuple(int(v) for v in pts["rs"])
                    cv2.line(frame, a, b, (0, 255, 0), 3)
                    cv2.putText(frame, f"shoulder width {sw:.0f} px -> {K[0, 0] * SHOULDER_M / max(sw, 1.0):.2f} m",
                                (min(a[0], b[0]), min(a[1], b[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                lost += 1

            err_px = err_mm = None
            if mk is not None:
                cv2.circle(frame, tuple(int(v) for v in mk[0]), 10, (255, 120, 0), 3)          # marker centre (blue)
                if raw is not None:
                    err_px = float(np.linalg.norm(raw[0] - mk[0]))
                    err_mm = err_px * marker_mm / max(mk[1], 1e-6)
                    cv2.line(frame, tuple(int(v) for v in mk[0]), tuple(int(v) for v in raw[0]), (255, 120, 0), 2)
                    cv2.putText(frame, f"error {err_mm:.0f} mm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 120, 0), 2)

            msg = "Tracking" if (state is not None and lost < HOLD_FRAMES) else "Stand facing the camera (shoulders visible)"
            cv2.putText(frame, msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"{fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            draw_background(tex, np.ascontiguousarray(frame))

            if state is not None and lost < HOLD_FRAMES:
                cx, cy, sw, roll = state
                Z = K[0, 0] * SHOULDER_M / max(sw, 1.0)          # distance from shoulder width
                X = (cx - K[0, 2]) * Z / K[0, 0]
                Y = (cy - K[1, 2]) * Z / K[1, 1]
                camera_projection()
                glLoadIdentity()
                glTranslatef(X, -Y, -Z)                          # OpenCV -> OpenGL axes
                glRotatef(-roll, 0, 0, 1)                        # follow shoulder tilt
                glRotatef(spin, 0, 1, 0)
                glRotatef(tilt, 1, 0, 0)
                k_ = scale * HEART_M / 2.0                       # model is ~2 units across
                glScalef(k_, k_, k_)
                glMultMatrixf(LPS_TO_VIEW)                       # anterior to camera, head up

                n_vis = sum(L["visible"] for L in layers)
                todo = []
                for L in layers:
                    if not L["visible"]:
                        continue
                    for p in L["parts"]:
                        if p["name"] in hidden:
                            continue
                        a = 1.0 if n_vis == 1 else L["alpha"]
                        if ghost:
                            a = min(a, GHOST_ALPHA)
                        if see_through and any(w in p["name"].lower() for w in SEE_THROUGH_WORDS):
                            a = min(a, 0.25)
                        todo.append((p, a))
                for p, a in todo:
                    if a >= 0.99:
                        draw(p, a)
                glDepthMask(GL_FALSE)
                for p, a in todo:
                    if a < 0.99:
                        draw(p, a)
                glDepthMask(GL_TRUE)

            if goggles is not None:
                px = glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE)
                goggles.show(np.frombuffer(px, np.uint8).reshape(H, W, 3)[::-1])
            pygame.display.flip()
            now = time.time()
            tracking = state is not None and lost < HOLD_FRAMES
            dist_m = (K[0, 0] * SHOULDER_M / max(state[2], 1.0)) if tracking else None
            log_rows.append([now - t_start, (now - prev) * 1000.0, (now - t_frame) * 1000.0,
                             int(raw is not None), int(tracking), getattr(finder, "n_people", 0),
                             None if raw is None else float(raw[0][0]), None if raw is None else float(raw[0][1]),
                             None if raw is None else float(raw[1]), None if raw is None else float(raw[2]),
                             dist_m,
                             None if mk is None else float(mk[0][0]), None if mk is None else float(mk[0][1]),
                             None if mk is None else mk[1], err_px, err_mm])
            frame_times.append(now - prev)
            fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6) if fps else 1 / max(now - prev, 1e-6)
            prev = now
            if fake and len(frame_times) > 120:
                return
    finally:
        cam.stop()
        if len(frame_times) > 10:
            ft = np.array(frame_times[5:])
            print(f"Average {1 / ft.mean():.1f} FPS over {len(ft)} frames")
        if len(log_rows) > 30:
            r = log_rows[10:]
            dt = np.array([x[1] for x in r]) / 1000.0
            f = 1.0 / dt
            proc = np.array([x[2] for x in r])
            det = np.mean([x[3] for x in r]) * 100
            line = (f"RESULT ar | frames {len(r)} | mean {len(dt) / dt.sum():.1f} FPS | SD {f.std():.1f} | min {f.min():.1f} | "
                    f"processing {proc.mean():.0f} +- {proc.std():.0f} ms | body detected {det:.0f}% of frames | max people {max(x[5] for x in r)}")
            errs = np.array([x[15] for x in r if x[15] is not None])
            if len(errs):
                dists = np.array([x[10] for x in r if x[15] is not None and x[10] is not None])
                line += f" | marker error {errs.mean():.1f} +- {errs.std():.1f} mm (n={len(errs)})"
                if len(dists):
                    line += f" at ~{dists.mean():.2f} m"
            print(line)
            if log_path:
                save_log(log_path, r, ["t_s", "frame_ms", "proc_ms", "body_detected", "tracking", "n_people",
                                       "heart_x_px", "heart_y_px", "shoulder_px", "roll_deg", "est_dist_m",
                                       "marker_x_px", "marker_y_px", "marker_side_px", "err_px", "err_mm"])
        pygame.quit()


if __name__ == "__main__":
    main()
