#!/usr/bin/env python3
"""
ar_heart.py - AR heart on an ArUco marker (Raspberry Pi 5 + Camera Module 3)
FYP-2  |  built from ar_cube.py + heart_viewer.py

Put this file in the same folder as heart_viewer.py and the export/ folder.

Usage (in a terminal on the Pi desktop / VNC):
    python3 ar_heart.py export/ct export/mr

Keys:
    1-9      show / hide a structure          C / M   CT / MRI layer
    T        see-through myocardium           + / -   bigger / smaller heart
    Left / Right arrows   spin heart on the marker
    R        reset size and spin              S       save screenshot
    Q / Esc  quit
"""
import os
import sys
import time

import cv2
import numpy as np

# reuse the loader, renderer and lighting from the viewer (sets the GLX fix too)
import heart_viewer as hv
from heart_viewer import load_layers, upload, draw, setup_gl, LPS_TO_VIEW, SEE_THROUGH_WORDS

import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *   # noqa: F401,F403

# ---- settings (same as ar_cube.py) --------------------------------------------
W, H = 1280, 720
MARKER_SIZE = 0.05        # metres: measure the BLACK square of your printed marker
MARKER_ID = None          # None = use the first marker seen, or set e.g. 0
HEART_SIZE = 1.5          # heart's largest dimension = HEART_SIZE x marker size
DETECT_SCALE = 0.5        # detect markers on a half-size image (faster)
SMOOTH = 0.6              # 0 = no smoothing, 0.9 = very smooth (more lag)
HOLD_FRAMES = 5           # keep the heart for a few frames if the marker is lost

# Approximate camera matrix for Camera Module 3 (replace after calibration)
fx = fy = 1000.0
K = np.array([[fx, 0, W / 2],
              [0, fy, H / 2],
              [0,  0,     1]], dtype=np.float32)
dist = np.zeros(5, dtype=np.float32)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
s = MARKER_SIZE / 2
marker_pts = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)

# OpenCV camera (x right, y down, z forward) -> OpenGL camera (y up, z back)
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


# ---- camera ----------------------------------------------------------------------
class PiCamera:
    def __init__(self):
        from picamera2 import Picamera2
        from libcamera import controls
        self.cam = Picamera2()
        cfg = self.cam.create_preview_configuration(main={"size": (W, H), "format": "RGB888"})
        self.cam.configure(cfg)
        self.cam.start()
        self.cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        time.sleep(1)

    def read(self):
        return self.cam.capture_array()        # BGR order for OpenCV

    def stop(self):
        self.cam.stop()


class FakeCamera:
    """For testing without a Pi: a marker that slowly tilts in front of a grey wall."""
    def __init__(self):
        self.t = 0
        self.marker = cv2.aruco.generateImageMarker(dictionary, 0, 200)
        self.marker = cv2.copyMakeBorder(self.marker, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    def read(self):
        self.t += 1
        img = np.full((H, W, 3), 120, np.uint8)
        a = 0.3 * np.sin(self.t / 15)
        c = np.array([W / 2, H / 2 + 80])
        w, h = 160, 160 * (0.75 + 0.15 * np.cos(a))
        dst = np.float32([c + [-w + 40 * a, -h], c + [w + 40 * a, -h], c + [w, h], c + [-w, h]])
        n = self.marker.shape[0]
        src = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
        warped = cv2.warpPerspective(cv2.cvtColor(self.marker, cv2.COLOR_GRAY2BGR),
                                     cv2.getPerspectiveTransform(src, dst), (W, H), borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(np.full((n, n), 255, np.uint8), cv2.getPerspectiveTransform(src, dst), (W, H))
        img[mask > 0] = warped[mask > 0]
        return img

    def stop(self):
        pass


# ---- pose ------------------------------------------------------------------------
def find_pose(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, None, fx=DETECT_SCALE, fy=DETECT_SCALE, interpolation=cv2.INTER_AREA)
    corners, ids, _ = detector.detectMarkers(small)
    if ids is None:
        return None
    for c, mid in zip(corners, ids.flatten()):
        if MARKER_ID is not None and mid != MARKER_ID:
            continue
        c = (c.reshape(4, 2) / DETECT_SCALE).astype(np.float32)
        cv2.cornerSubPix(gray, c, (5, 5), (-1, -1),            # refine corners on the full-size image
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01))
        ok, rvec, tvec = cv2.solvePnP(marker_pts, c, K, dist,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            return rvec.ravel(), tvec.ravel(), c, int(mid)
    return None


# ---- drawing ---------------------------------------------------------------------
def make_background_texture():
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_BGR, GL_UNSIGNED_BYTE, None)
    return tex


def draw_background(tex, frame):
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_BGR, GL_UNSIGNED_BYTE, frame)
    glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST); glEnable(GL_TEXTURE_2D)
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, 1, 0, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW); glLoadIdentity()
    glColor4f(1, 1, 1, 1)
    glDisableClientState(GL_VERTEX_ARRAY); glDisableClientState(GL_NORMAL_ARRAY)
    glBegin(GL_QUADS)                       # image row 0 is the top of the screen
    glTexCoord2f(0, 1); glVertex2f(0, 0)
    glTexCoord2f(1, 1); glVertex2f(1, 0)
    glTexCoord2f(1, 0); glVertex2f(1, 1)
    glTexCoord2f(0, 0); glVertex2f(0, 1)
    glEnd()
    glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_NORMAL_ARRAY)
    glDisable(GL_TEXTURE_2D); glEnable(GL_LIGHTING); glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)


def camera_projection(near=0.01, far=10.0):
    """OpenGL projection that matches the camera matrix K."""
    l = -K[0, 2] * near / K[0, 0]
    r = (W - K[0, 2]) * near / K[0, 0]
    b = -(H - K[1, 2]) * near / K[1, 1]
    t = K[1, 2] * near / K[1, 1]
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glFrustum(l, r, b, t, near, far)
    glMatrixMode(GL_MODELVIEW)


def main():
    paths = [p for p in sys.argv[1:] if not p.startswith("--")] or ["export/ct", "export/mr"]
    fake = "--fake" in sys.argv
    layers = load_layers(paths)
    structures = []
    for L in layers:
        for p in L["parts"]:
            if p["name"] not in structures:
                structures.append(p["name"])
    # depth of the model front-to-back (LPS y), used to sit it on the marker
    allv = np.concatenate([p["v"] for L in layers for p in L["parts"]])
    depth = float(allv[:, 1].max() - allv[:, 1].min())
    print(__doc__.split("Keys:")[1])
    for i, s_ in enumerate(structures):
        print(f"  [{i + 1}] {s_}")

    pygame.init()
    pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("AR heart")
    setup_gl(W, H)
    upload(layers)
    tex = make_background_texture()
    cam = FakeCamera() if fake else PiCamera()

    hidden, see_through = set(), False
    size, spin = HEART_SIZE, 0.0
    pose, lost = None, 99
    fps, prev, shot = 0.0, time.time(), 0
    frame_times = []

    try:
        while True:
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
                    elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        size *= 1.15
                    elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        size /= 1.15
                    elif k == pygame.K_r:
                        size, spin = HEART_SIZE, 0.0
                    elif k == pygame.K_s:
                        shot += 1
                        data = glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE)
                        img = pygame.image.fromstring(data, (W, H), "RGB", True)
                        name = f"ar_heart_{shot}.png"
                        pygame.image.save(img, name)
                        print("Saved", name)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                spin -= 2
            if keys[pygame.K_RIGHT]:
                spin += 2

            frame = cam.read()
            found = find_pose(frame)
            if found:
                rvec, tvec, corners, mid = found
                if pose is not None and lost < HOLD_FRAMES and np.linalg.norm(tvec - pose[1]) < 0.05:
                    rvec = SMOOTH * pose[0] + (1 - SMOOTH) * rvec      # simple smoothing
                    tvec = SMOOTH * pose[1] + (1 - SMOOTH) * tvec
                pose, lost = (rvec, tvec), 0
            else:
                lost += 1

            # text on the camera image
            if pose is not None and lost < HOLD_FRAMES:
                cv2.putText(frame, f"{np.linalg.norm(pose[1]) * 100:.0f} cm", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Show the marker to the camera", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"{fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            draw_background(tex, np.ascontiguousarray(frame))

            if pose is not None and lost < HOLD_FRAMES:
                camera_projection()
                R, _ = cv2.Rodrigues(pose[0])
                M = np.eye(4); M[:3, :3] = R; M[:3, 3] = pose[1]
                glLoadMatrixf((CV_TO_GL @ M).T.astype(np.float32))      # marker -> camera
                k_ = size * MARKER_SIZE / 2.0                            # model is ~2 units across
                glTranslatef(0, 0, k_ * depth / 2)                       # sit on the marker
                glRotatef(spin, 0, 0, 1)
                glScalef(k_, k_, k_)
                glMultMatrixf(LPS_TO_VIEW)        # anterior faces the camera, head = marker top

                n_vis = sum(L["visible"] for L in layers)
                todo = []
                for L in layers:
                    if not L["visible"]:
                        continue
                    for p in L["parts"]:
                        if p["name"] in hidden:
                            continue
                        a = 1.0 if n_vis == 1 else L["alpha"]
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

            pygame.display.flip()

            now = time.time()
            frame_times.append(now - prev)
            fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6) if fps else 1 / max(now - prev, 1e-6)
            prev = now
            if fake and len(frame_times) > 40:
                return
    finally:
        cam.stop()
        if len(frame_times) > 10:
            ft = np.array(frame_times[5:])
            print(f"Average {1 / ft.mean():.1f} FPS over {len(ft)} frames")
        pygame.quit()


if __name__ == "__main__":
    main()
