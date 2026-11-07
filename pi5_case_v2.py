"""
Raspberry Pi 5 desk case v2 with a tilting Camera Module 3 pod on the lid.  Units: mm.
Parts: base, lid (with hinge lugs + ribbon slot + hex vents), camera pod, pod cap.
Pi 5 dimensions from the official product brief: board 85 x 56, holes 58 x 49 @ 3.5 mm,
USB-C 11.2, micro-HDMI 25.8 / 39.2 (from the left on the long edge),
Ethernet 10.2, USB 29.1 / 47 (from the HDMI edge on the short edge).
Camera Module 3: 25 x 24 mm PCB, held in side rails (no screw holes needed).
Hardware: 4x M2.5 x 6 screws (Pi), 1x M3 x 20 bolt + nut (pod hinge).
"""
import numpy as np, trimesh
from manifold3d import Manifold

SEG = 48
def box(x0, y0, z0, x1, y1, z1):
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
def cyl_z(x, y, z0, h, d, seg=SEG):
    return Manifold.cylinder(h, d / 2, d / 2, seg).translate([x, y, z0])
def cyl_x(x0, y, z, h, d):
    return Manifold.cylinder(h, d / 2, d / 2, SEG).rotate([0, 90, 0]).translate([x0, y, z])
def cyl_y(x, y0, z, h, d):
    return Manifold.cylinder(h, d / 2, d / 2, SEG).rotate([-90, 0, 0]).translate([x, y0, z])
def rbox(x0, y0, z0, x1, y1, z1, r, top_r=0.0):
    """Box with vertical edges rounded (r) and optionally the top edges rounded (top_r)."""
    pts = []
    for x in (x0 + r, x1 - r):
        for y in (y0 + r, y1 - r):
            pts.append(cyl_z(x, y, z0, (z1 - z0) - top_r, 2 * r))
            if top_r > 0:
                pts.append(Manifold.sphere(r, SEG).scale([1, 1, top_r / r]).translate([x, y, z1 - top_r]))
    return Manifold.batch_hull(pts)
def rslot(x0, y0, z0, x1, y1, z1, r):   # rounded window cut along Y or X (rounded in the cut plane)
    return rbox(x0, y0, z0, x1, y1, z1, min(r, (x1 - x0) / 2 - 0.01, (y1 - y0) / 2 - 0.01))
def win_y(xc, w, z0, z1, wall_y0, wall_y1, r=1.5):    # window through a wall normal to Y
    pts = [cyl_y(x, wall_y0, z, wall_y1 - wall_y0, 2 * r) for x in (xc - w / 2 + r, xc + w / 2 - r) for z in (z0 + r, z1 - r)]
    return Manifold.batch_hull(pts)
def win_x(yc, w, z0, z1, wall_x0, wall_x1, r=1.5):
    pts = [cyl_x(wall_x0, y, z, wall_x1 - wall_x0, 2 * r) for y in (yc - w / 2 + r, yc + w / 2 - r) for z in (z0 + r, z1 - r)]
    return Manifold.batch_hull(pts)

# ============ base ============
T, R = 2.2, 7.0                     # wall, outer corner radius
PX, PY = 1.0, 1.5                   # Pi board corner inside the interior
IX, IY = 87.0, 59.0                 # interior (board 85 x 56 + clearance; ports overhang into +X wall)
STAND, PCB = 5.0, 1.6
TOPZ = STAND + PCB                  # top of PCB
IZ = 28.0                           # interior height: USB stack 16 mm above PCB + lid lip 4 mm clear
outer = rbox(-T, -T, -T, IX + T, IY + T, IZ, R)
inner = rbox(0, 0, 0, IX, IY, IZ + 1, R - T)
base = outer - inner
holes = [(PX + 3.5, PY + 3.5), (PX + 61.5, PY + 3.5), (PX + 3.5, PY + 52.5), (PX + 61.5, PY + 52.5)]
for (x, y) in holes:
    base = base + cyl_z(x, y, 0, STAND, 6.5) - cyl_z(x, y, -T - 1, STAND + T + 2, 2.4)
# -Y wall: USB-C + 2x micro-HDMI (individual rounded windows sized for plug overmoulds)
for xc, w in ((PX + 11.2, 12.5), (PX + 25.8, 11.5), (PX + 39.2, 11.5)):
    base = base - win_y(xc, w, TOPZ - 2.0, TOPZ + 6.0, -T - 1, 1.0)
# +X wall: Ethernet + 2x USB
for yc, w, h in ((PY + 10.2, 17.0, 14.5), (PY + 29.1, 15.5, 17.0), (PY + 47.0, 15.5, 17.0)):
    base = base - win_x(yc, w, TOPZ - 0.6, TOPZ + h, IX - 1, IX + T + 1)
# -X wall: microSD slot (card is under the board at the -X edge) + finger notch
base = base - win_x(PY + 28.0, 15.0, STAND - 3.2, STAND + 0.6, -T - 1, 1.0, r=1.0)
# side vent slots on the +Y wall (pill shapes)
for i in range(6):
    xc = 18 + i * 9
    base = base - win_y(xc, 4.0, 8.0, 19.0, IY - 1, IY + T + 1, r=1.9)
# rubber-foot recesses
for (x, y) in [(7, 7), (IX - 7, 7), (7, IY - 7), (IX - 7, IY - 7)]:
    base = base - cyl_z(x, y, -T - 0.1, 0.8, 10)

# ============ lid ============
LT, CLR, LIP_H, LIP_W = 2.4, 0.3, 4.0, 1.6
plate = rbox(-T, -T, 0, IX + T, IY + T, LT, R, top_r=1.6)
lip = rbox(CLR, CLR, -LIP_H, IX - CLR, IY - CLR, 0.01, R - T - CLR) - rbox(CLR + LIP_W, CLR + LIP_W, -LIP_H - 1, IX - CLR - LIP_W, IY - CLR - LIP_W, 0.2, max(R - T - CLR - LIP_W, 0.5))
lid = plate + lip
# hex vent field over the SoC / RAM area
HEX = 5.2
for i in range(-3, 4):
    for j in range(-2, 3):
        x = PX + 33 + i * 6.2 + (3.1 if j % 2 else 0)
        y = PY + 31 + j * 5.4
        lid = lid - Manifold.cylinder(LT + 2, HEX / 2, HEX / 2, 6).translate([x, y, -1])
# GPIO slot (jumper wires / LCD cable)
lid = lid - rbox(PX + 6.0, PY + 48.5, -LIP_H - 1, PX + 59.0, PY + 56.0, LT + 1, 1.5)
# camera ribbon slot + hinge lugs (pod sits above the CSI connectors near the HDMI edge)
HX, HY, HZ = PX + 54.0, PY + 13.0, LT + 6.0     # hinge axis (along X) position
lid = lid - rbox(HX - 9.5, HY - 10.3, -LIP_H - 1, HX + 9.5, HY - 5.4, LT + 1, 1.2)   # ribbon slot under the pod
for side in (-1, 1):
    x0 = HX + side * 8.6 - (0 if side > 0 else 3.4)
    lug = Manifold.batch_hull([box(x0, HY - 5, LT - 0.5, x0 + 3.4, HY + 5, LT + 0.5),
                               cyl_x(x0, HY, HZ, 3.4, 10.0)])
    lid = lid + lug - cyl_x(x0 - 1, HY, HZ, 5.4, 3.3)

# ============ camera pod ============
# pod frame: x across, y = viewing direction (+Y), z up. Hinge axis at origin.
CW, CD = 22.0, 11.5           # cavity width (narrower than PCB), cavity depth
RW, RT = 25.9, 1.4            # rail slot width (PCB 25) and thickness (PCB ~1)
PCB_Y = 5.0                   # PCB front face distance behind the front wall
PW = 2.0                      # pod wall
H_CAV = 25.0                  # cavity height (PCB 24)
Z0 = 5.5                      # cavity floor height above hinge axis
pod_outer = rbox(-RW / 2 - PW, -PW - CD + 0.0, 0.0, RW / 2 + PW, PW, Z0 + H_CAV, 3.0, top_r=0.0)
# y: front wall at y in [0, PW]; cavity y in [-CD, 0]; camera looks toward +Y
pod = pod_outer - box(-CW / 2, -CD, Z0, CW / 2, 0.0, Z0 + H_CAV + 1)
pod = pod - box(-RW / 2, -PCB_Y - RT, Z0, RW / 2, -PCB_Y, Z0 + H_CAV + 1)        # rails
# lens window (generous: covers Camera Module 3 lens anywhere in the upper-middle area)
pod = pod - Manifold.batch_hull([cyl_y(0, -1, Z0 + H_CAV - 7.2, PW + 2, 13.0), cyl_y(0, -1, Z0 + H_CAV - 13.0, PW + 2, 13.0)])
# ribbon exit through the floor behind the PCB
pod = pod - rbox(-9.0, -CD + 0.2, -1, 9.0, -PCB_Y - RT - 0.2, Z0 + 0.5, 1.0)
# hinge boss under the front of the pod (between the lid lugs)
boss = Manifold.batch_hull([box(-8.2, -5.0, Z0 - 0.5, 8.2, PW, Z0 + 0.5), cyl_x(-8.2, -1.5, 0.0, 16.4, 10.0)])
pod = (pod + boss) - cyl_x(-9, -1.5, 0.0, 18, 3.3)
# cap that plugs into the top of the cavity and keeps the camera in its rails
cap = rbox(-RW / 2 - PW, -PW - CD, 0, RW / 2 + PW, PW, 2.0, 3.0, top_r=1.0) + box(-CW / 2 + 0.2, -CD + 0.2, -3.0, CW / 2 - 0.2, -0.2, 0.01) \
      - box(-CW / 2 + 1.8, -CD + 1.8, -4, CW / 2 - 1.8, -1.8, -0.5)
cap = cap - box(-RW / 2 - 0.2, -PCB_Y - RT - 0.2, -4, RW / 2 + 0.2, -PCB_Y + 0.2, -0.6)   # clears PCB top edge

def tm(m):
    mesh = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=False)
def save(m, name, flip=False):
    t = tm(m)
    if flip:
        t.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    t.apply_translation(-t.bounds[0]); t.export(name)
    print(f"{name:28s} watertight={t.is_watertight}  size={np.round(t.extents, 1)} mm")
    return t

if __name__ == "__main__":
    save(base, "v2_base.stl")
    save(lid, "v2_lid.stl")                       # print upright (lugs up), lip needs no support
    save(pod, "v2_camera_pod.stl")
    save(cap, "v2_pod_cap.stl", flip=True)
    # assembled model for preview
    asm_lid = lid.translate([0, 0, IZ])
    asm_pod = pod.translate([HX, HY + 1.5, IZ + HZ])
    asm_cap = cap.translate([HX, HY + 1.5, IZ + HZ + Z0 + H_CAV])
    tm(base).export("v2_assembly_base.stl"); tm(asm_lid).export("v2_assembly_lid.stl")
    tm(asm_pod).export("v2_assembly_pod.stl"); tm(asm_cap).export("v2_assembly_cap.stl")
