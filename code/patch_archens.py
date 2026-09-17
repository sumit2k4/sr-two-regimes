"""Add the conventional architecture-ensemble baseline (three different CNN
backbones averaged on the super-resolved domain), which is the ensemble design
the literature and the project proposal actually describe, and which QGE must be
measured against."""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ qge.py
P = os.path.join(HERE, "qge.py")
s = io.open(P, encoding="utf-8").read()

anchor = 'def _fit_global_weights(Pv, yv, grid=21):'
new_fn = '''def load_arch_ensemble(ds, seed, protocol, view="hfgan_g",
                       backbones=("resnet18", "efficientnet_b0", "densenet121")):
    """Posteriors of several architectures on the SAME super-resolved view: the
    conventional architecture ensemble. Returns None if the runs are absent."""
    Pv, Pt = [], []
    for b in backbones:
        f = os.path.join(CLS, ds + "_" + view + "_p1_" + b + "_" + str(seed) + ".npz")
        if not os.path.exists(f):
            return None, None
        z = np.load(f)
        Pv.append(softmax_np(z["val_" + view + "_" + protocol]))
        Pt.append(softmax_np(z["test_" + view + "_" + protocol]))
    return np.stack(Pv, 1), np.stack(Pt, 1)


'''
assert anchor in s
s = s.replace(anchor, new_fn + anchor)

anchor = '''        m_st, p_st = _stacking(Pv, yv, Pt, yt, ncls)'''
add = '''        Av, At = load_arch_ensemble(ds, seed, protocol)
        if At is not None:
            preds["arch_ens"] = At.mean(1)
            res["arch_ens"] = metrics(preds["arch_ens"], yt, ncls)
        m_st, p_st = _stacking(Pv, yv, Pt, yt, ncls)'''
assert anchor in s
s = s.replace(anchor, add)
io.open(P, "w", encoding="utf-8", newline="\n").write(s)

# ----------------------------------------------------------- make_tables.py
P = os.path.join(HERE, "make_tables.py")
s = io.open(P, encoding="utf-8").read()
old = '''            ("confmax", "Confidence-max routing"),'''
new = '''            ("confmax", "Confidence-max routing"),
            ("arch_ens", "Architecture ensemble (3 CNNs, SR view)"),'''
assert old in s
s = s.replace(old, new)
old = '''        for k, lab in QGE_KEYS:
            rows.append('''
new = '''        for k, lab in QGE_KEYS:
            if k not in blk[list(blk)[0]]:
                continue
            rows.append('''
assert old in s
s = s.replace(old, new)
io.open(P, "w", encoding="utf-8", newline="\n").write(s)

# --------------------------------------------------------------- figures.py
P = os.path.join(HERE, "figures.py")
s = io.open(P, encoding="utf-8").read()
old = '''    keys = ["single_bic", "single_edsr", "single_hfgan_g", "majority", "avg",
            "global_w", "stacking", "confmax", "qge"]
    lbl = ["Bicubic", "EDSR", "HFGAN-G", "Majority", "Average", "Weighted",
           "Stacking", "Conf-max", "QGE (ours)"]'''
new = '''    keys = ["single_bic", "single_edsr", "single_hfgan_g", "majority", "avg",
            "global_w", "stacking", "confmax", "arch_ens", "qge"]
    lbl = ["Bicubic", "EDSR", "HFGAN-G", "Majority", "Average", "Weighted",
           "Stacking", "Conf-max", "Arch. ens.", "QGE (ours)"]
    probe = T[dss[0]]["qge_p1"]
    probe = probe[list(probe)[0]]
    keep = [i for i, k in enumerate(keys) if k in probe]
    keys = [keys[i] for i in keep]
    lbl = [lbl[i] for i in keep]'''
assert old in s
s = s.replace(old, new)
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("architecture-ensemble baseline wired into qge / make_tables / figures")
