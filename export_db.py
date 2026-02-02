# ========= SAVE SIDE (Jupyter) =========
# Save the trained model to a folder (tree + index + meta), converting cv2.KeyPoint to tuples.
# Usage after you finish training/finalizing:
#   save_vt_model(db, out_dir="vocab_db", include_meta=True)

import os, io, gzip, pickle, tempfile
import numpy as np
import cv2

def save_vt_model(db, out_dir, include_meta=True):
    # ---- helpers ----
    def _atomic_write(path, data_bytes):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as tmp:
            tmp.write(data_bytes); tmp.flush(); os.fsync(tmp.fileno())
            tmp_path = tmp.name
        os.replace(tmp_path, path)

    def _bytes_dump_gz(obj) -> bytes:
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=6) as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        return buf.getvalue()

    def _iter_nodes_preorder(root):
        stack, seen = [root], set()
        while stack:
            n = stack.pop()
            if id(n) in seen: continue
            seen.add(id(n)); yield n
            for c in reversed(getattr(n, "children", []) or []):
                stack.append(c)

    def export_tree_dict(tree):
        assert tree is not None and tree.root is not None, "Tree must be trained."
        nodes = []
        for n in _iter_nodes_preorder(tree.root):
            nodes.append({
                "node_id": int(n.node_id),
                "is_leaf": bool(getattr(n, "is_leaf", False)),
                "centroid": (None if getattr(n, "centroid", None) is None
                             else np.ascontiguousarray(n.centroid, dtype=np.float32)),
                "children_ids": [int(c.node_id) for c in (getattr(n, "children", None) or [])],
            })
        leaf_id_map = getattr(tree, "leaf_id_map", None)
        if leaf_id_map is not None:
            leaf_id_map = {int(k): int(v) for k, v in leaf_id_map.items()}
        return {
            "version": 1,
            "k": int(getattr(tree, "k", 10)),
            "L": int(getattr(tree, "L", 6)),
            "root_id": int(tree.root.node_id),
            "nodes": nodes,
            "leaf_id_map": leaf_id_map,
        }

    def export_index_dict(index):
        assert index is not None, "Index is None; finalize() before saving."
        return {
            "version": 1,
            "num_leaves": int(index.num_leaves),
            "doc_tf": {int(d): {int(w): int(tf) for w, tf in tfmap.items()}
                       for d, tfmap in index.doc_tf.items()},
            "idf": np.ascontiguousarray(index.idf, dtype=np.float32),
            "doc_norm": {int(d): float(v) for d, v in index.doc_norm.items()},
            "N": int(getattr(index, "N", len(index.doc_tf))),
            "ext2int": {k: int(v) for k, v in getattr(index, "ext2int", {}).items()},
            "int2ext": list(getattr(index, "int2ext", [])),
            "stop_leaves": set(getattr(index, "stop_leaves", set())),
        }

    def _kp_to_tuple(kp):
        return (float(kp.pt[0]), float(kp.pt[1]),
                float(kp.size), float(kp.angle),
                float(kp.response), int(kp.octave), int(kp.class_id))

    def export_meta_dict(db, include=True):
        if not include:
            return None
        meta = getattr(db, "image_meta", {}) or {}
        # sanitize kps → tuples to avoid "cannot pickle 'cv2.KeyPoint'"
        clean = {}
        for img_id, m in meta.items():
            kps = m.get("kps", None)
            if kps and len(kps) and isinstance(kps[0], cv2.KeyPoint):
                kps = [_kp_to_tuple(k) for k in kps]
            clean[img_id] = {
                "path": m.get("path"),
                "descs": (np.ascontiguousarray(m.get("descs"), dtype=np.float32)
                          if m.get("descs") is not None else None),
                "kps": kps
            }
        return {"image_meta": clean}

    # ---- save ----
    os.makedirs(out_dir, exist_ok=True)
    tree_dict = export_tree_dict(db.tree)
    idx_dict  = export_index_dict(db.index)
    meta_dict = export_meta_dict(db, include=include_meta)

    _atomic_write(os.path.join(out_dir, "tree.pkl.gz"),  _bytes_dump_gz(tree_dict))
    _atomic_write(os.path.join(out_dir, "index.pkl.gz"), _bytes_dump_gz(idx_dict))
    if meta_dict is not None:
        _atomic_write(os.path.join(out_dir, "meta.pkl.gz"),  _bytes_dump_gz(meta_dict))
    print(f"[save_vt_model] Saved to: {out_dir}")

