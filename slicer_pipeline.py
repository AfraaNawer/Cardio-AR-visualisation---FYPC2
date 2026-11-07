# ===================================================================
# FYP-2 Slicer pipeline: MM-WHS CT + MRI  ->  aligned 3D heart models
# Paste into 3D Slicer: View -> Python Console, then press Enter.
#
#  1. Loads CT + MRI images and their expert labels from
#     Downloads/heartdataset
#  2. Turns the labels into named, coloured 3D segments (7 structures)
#  3. Aligns the MRI heart onto the CT heart (similarity transform
#     from the centres of the 7 structures) and prints the error
#  4. Exports one STL per structure to Downloads/heartdataset/export/ct
#     and .../export/mr  (ready for the Raspberry Pi viewer)
# ===================================================================
import os, glob, vtk, slicer, numpy as np

DATA = os.path.join(os.path.expanduser('~'), 'Downloads', 'heartdataset')
OUT = os.path.join(DATA, 'export')

# MM-WHS label values -> (name, colour)
LABELS = {
    205: ('LV myocardium',      (0.85, 0.55, 0.40)),
    420: ('Left atrium',        (0.95, 0.35, 0.35)),
    500: ('LV blood cavity',    (0.80, 0.10, 0.15)),
    550: ('Right atrium',       (0.35, 0.55, 0.95)),
    600: ('RV blood cavity',    (0.15, 0.30, 0.85)),
    820: ('Ascending aorta',    (0.95, 0.75, 0.20)),
    850: ('Pulmonary artery',   (0.30, 0.75, 0.85)),
}

def find(pattern):
    hits = sorted(glob.glob(os.path.join(DATA, pattern)))
    if not hits:
        raise FileNotFoundError('Not found in %s: %s' % (DATA, pattern))
    return hits[0]

slicer.mrmlScene.Clear(0)

# ---------- colour table with the structure names -------------------
colors = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLColorTableNode', 'MMWHS labels')
colors.SetTypeToUser()
colors.SetNumberOfColors(851)
colors.SetColor(0, 'background', 0, 0, 0, 0)
for v, (name, rgb) in LABELS.items():
    colors.SetColor(v, name, rgb[0], rgb[1], rgb[2], 1.0)

def centroids(labelNode):
    """Centre (RAS, mm) of every labelled structure."""
    arr = slicer.util.arrayFromVolume(labelNode)          # K, J, I
    m = vtk.vtkMatrix4x4(); labelNode.GetIJKToRASMatrix(m)
    M = slicer.util.arrayFromVTKMatrix(m)
    out = {}
    for v in LABELS:
        kji = np.argwhere(arr == v)
        if len(kji):
            ijk1 = np.c_[kji[:, ::-1], np.ones(len(kji))].mean(0)
            out[v] = (M @ ijk1)[:3]
    return out

def load_modality(tag, image_pat, label_pat):
    img = slicer.util.loadVolume(find(image_pat), {'name': tag + ' image'})
    lab = slicer.util.loadLabelVolume(find(label_pat), {'name': tag + ' label'})
    lab.GetDisplayNode().SetAndObserveColorNodeID(colors.GetID())
    seg = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', tag + ' heart')
    seg.CreateDefaultDisplayNodes()
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lab, seg)
    seg.SetReferenceImageGeometryParameterFromVolumeNode(img)
    print('%s: loaded %d structures' % (tag, seg.GetSegmentation().GetNumberOfSegments()))
    return img, lab, seg

print('Loading data from', DATA, '...')
ctImg, ctLab, ctSeg = load_modality('CT', 'ct_train_*_image*.nii*', 'ct_train_*_label*.nii*')
mrImg, mrLab, mrSeg = load_modality('MR', 'mr_train_*_image*.nii*', 'mr_train_*_label*.nii*')

# ---------- align MRI -> CT (similarity: rotation + translation + scale)
c_ct, c_mr = centroids(ctLab), centroids(mrLab)
common = [v for v in LABELS if v in c_ct and v in c_mr]
P = np.array([c_mr[v] for v in common]); Q = np.array([c_ct[v] for v in common])
mp, mq = P.mean(0), Q.mean(0)
U, S, Vt = np.linalg.svd((Q - mq).T @ (P - mp))
D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
R = U @ D @ Vt
s = np.trace(np.diag(S) @ D) / ((P - mp) ** 2).sum()
t = mq - s * R @ mp
T = np.eye(4); T[:3, :3] = s * R; T[:3, 3] = t
res = np.linalg.norm((s * (R @ P.T)).T + t - Q, axis=1)

tnode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode', 'MR to CT')
slicer.util.updateTransformMatrixFromArray(tnode, T)
for n in (mrImg, mrLab, mrSeg):
    n.SetAndObserveTransformNodeID(tnode.GetID())

print('\nMRI -> CT alignment (%d landmarks)' % len(common))
print('  scale factor : %.3f' % s)
for v, r in zip(common, res):
    print('  %-18s residual %5.1f mm' % (LABELS[v][0], r))
print('  RMS error    : %.1f mm' % np.sqrt((res ** 2).mean()))

# ---------- build smooth surfaces and export ----------------------------
def export(seg, folder):
    seg.GetSegmentation().SetConversionParameter('Smoothing factor', '0.4')
    seg.GetSegmentation().SetConversionParameter('Decimation factor', '0.7')
    seg.CreateClosedSurfaceRepresentation()
    os.makedirs(folder, exist_ok=True)
    for f in glob.glob(os.path.join(folder, '*.stl')):
        os.remove(f)
    total = 0
    sg = seg.GetSegmentation()
    for i in range(sg.GetNumberOfSegments()):
        poly = vtk.vtkPolyData()
        seg.GetClosedSurfaceRepresentation(sg.GetNthSegmentID(i), poly)
        total += poly.GetNumberOfCells()
    slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsClosedSurfaceRepresentationToFiles(
        folder, seg, None, 'STL', True, 1.0, False)
    print('  %-9s %8d triangles -> %s' % (seg.GetName(), total, folder))

print('\nBuilding 3D surfaces (can take a minute or two)...')
export(ctSeg, os.path.join(OUT, 'ct'))
slicer.vtkSlicerTransformLogic().hardenTransform(mrSeg)   # bake MR into CT space
export(mrSeg, os.path.join(OUT, 'mr'))

# ---------- show the result -------------------------------------------
mrSeg.GetDisplayNode().SetOpacity3D(0.35)
slicer.util.setSliceViewerLayers(background=ctImg)
slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()
print('\nDONE. Files:')
for sub in ('ct', 'mr'):
    print(' ', sub, sorted(os.listdir(os.path.join(OUT, sub))))
