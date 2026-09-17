"""Add ConvNeXt-Tiny as the primary branch backbone.

ConvNeXt is chosen over Swin-T because it is fully convolutional and therefore
resolution agnostic, which the study needs: inputs range from 16x16 (EuroSAT LR)
to 256x256 (UC Merced HR). Swin-T is also unavailable in torchvision 0.12.

ConvNeXt's stem is stride 4 followed by three stride-2 stages, so the smallest
input it can process is 32x32. The EuroSAT LR-native domain (16x16) is therefore
out of reach for this backbone; train_cls.py now skips such runs cleanly instead
of crashing.
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------ cls_models.py
p = os.path.join(HERE, "cls_models.py")
s = io.open(p, encoding="utf-8").read()
if "convnext_tiny" not in s:
    s = s.replace(
        'BACKBONES = ["resnet18", "efficientnet_b0", "densenet121"]',
        'BACKBONES = ["convnext_tiny", "resnet18", "efficientnet_b0", "densenet121"]\n'
        '\n'
        '# smallest square input each backbone can ingest\n'
        'MIN_INPUT = {"convnext_tiny": 32, "convnext_small": 32, "resnet18": 8,\n'
        '             "efficientnet_b0": 16, "densenet121": 32}')
    s = s.replace(
        '    elif name == "efficientnet_b0":',
        '    elif name.startswith("convnext"):\n'
        '        m = getattr(tvm, name)(pretrained=pretrained)\n'
        '        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_classes)\n'
        '    elif name == "efficientnet_b0":')
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)
    print("cls_models.py: ConvNeXt added")

# ------------------------------------------------------------ train_cls.py
p = os.path.join(HERE, "train_cls.py")
s = io.open(p, encoding="utf-8").read()

if "MIN_INPUT" not in s:
    s = s.replace("from cls_models import build_cls, IMNET_MEAN, IMNET_STD",
                  "from cls_models import build_cls, IMNET_MEAN, IMNET_STD, MIN_INPUT")

    # refuse to train a backbone on an input it cannot process
    old = """    X = np.load(dom_path(a.dataset, a.domain))
    tr = GPUBatcher(X, labels, sp["train"], bs, True, dev)"""
    new = """    X = np.load(dom_path(a.dataset, a.domain))
    need = MIN_INPUT.get(a.backbone, 8)
    if X.shape[1] < need:
        print("SKIP %s: %s needs >=%dpx, domain is %dpx"
              % (tag, a.backbone, need, X.shape[1]))
        return
    tr = GPUBatcher(X, labels, sp["train"], bs, True, dev)"""
    assert old in s
    s = s.replace(old, new)

    # skip evaluation domains that are too small for this backbone
    old = """        Y = np.load(dom_path(a.dataset, d), mmap_mode="r")
        if Y.shape[1] != base_shape and not (d.startswith("lr_") or a.domain.startswith("lr_")):
            continue"""
    new = """        Y = np.load(dom_path(a.dataset, d), mmap_mode="r")
        if Y.shape[1] < need:
            del Y
            continue
        if Y.shape[1] != base_shape and not (d.startswith("lr_") or a.domain.startswith("lr_")):
            continue"""
    assert old in s
    s = s.replace(old, new)

    # ConvNeXt fine-tunes far better with AdamW than with SGD
    old = """    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, epochs=a.epochs,
                                              steps_per_epoch=len(tr), pct_start=0.25)"""
    new = """    if a.backbone.startswith("convnext"):
        peak = a.lr if a.lr != 0.01 else 3e-4
        opt = torch.optim.AdamW(model.parameters(), lr=peak, weight_decay=0.05)
    else:
        peak = a.lr
        opt = torch.optim.SGD(model.parameters(), lr=peak, momentum=0.9,
                              weight_decay=5e-4, nesterov=True)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=peak, epochs=a.epochs,
                                              steps_per_epoch=len(tr), pct_start=0.25)"""
    assert old in s
    s = s.replace(old, new)
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)
    print("train_cls.py: min-input guard + AdamW for ConvNeXt")

# ------------------------------------------------------------- common.py
p = os.path.join(HERE, "common.py")
s = io.open(p, encoding="utf-8").read()
if "PRIMARY_BACKBONE" not in s:
    s = s.replace("SCALE = 4",
                  'SCALE = 4\n\n'
                  '# backbone whose results are reported in the manuscript; the\n'
                  '# ResNet-18 runs are retained as a backbone-robustness check\n'
                  'PRIMARY_BACKBONE = os.environ.get("P3_BACKBONE", "convnext_tiny")')
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)
    print("common.py: PRIMARY_BACKBONE added")

# ------------------- analysis modules read the primary backbone -------------
for name, old, new in [
    ("analyze.py", 'def table_factorial(ds, backbone="resnet18"):',
     'def table_factorial(ds, backbone=None):\n    backbone = backbone or C.PRIMARY_BACKBONE'),
    ("analyze.py", 'def table_confound(ds, backbone="resnet18"):',
     'def table_confound(ds, backbone=None):\n    backbone = backbone or C.PRIMARY_BACKBONE'),
    ("analyze.py", 'def table_perclass(ds, model="hfgan_g", backbone="resnet18"):',
     'def table_perclass(ds, model="hfgan_g", backbone=None):\n    backbone = backbone or C.PRIMARY_BACKBONE'),
    ("analyze.py", 'def table_quality_vs_gain(ds, backbone="resnet18"):',
     'def table_quality_vs_gain(ds, backbone=None):\n    backbone = backbone or C.PRIMARY_BACKBONE'),
    ("analyze_transfer.py", 'BACKBONE = "resnet18"', 'BACKBONE = C.PRIMARY_BACKBONE'),
    ("qge.py", 'def load_branches(ds, seed, protocol, backbone="resnet18"):',
     'def load_branches(ds, seed, protocol, backbone=None):\n    backbone = backbone or C.PRIMARY_BACKBONE'),
]:
    p = os.path.join(HERE, name)
    s = io.open(p, encoding="utf-8").read()
    if old in s:
        s = s.replace(old, new, 1)
        io.open(p, "w", encoding="utf-8", newline="\n").write(s)
        print(name, "->", old.split("(")[0][:40])

# the architecture ensemble should use three genuinely different families
p = os.path.join(HERE, "qge.py")
s = io.open(p, encoding="utf-8").read()
s = s.replace('backbones=("resnet18", "efficientnet_b0", "densenet121")',
              'backbones=("convnext_tiny", "efficientnet_b0", "densenet121")')
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("qge.py: architecture ensemble uses ConvNeXt/EffNet/DenseNet")
print("done")
