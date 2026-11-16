"""Worker process for panel_uv.unwrap_charts (run as a plain script, not through multiprocessing).

Started by panel_uv with the same Python as ComfyUI. Reads length-prefixed pickles on stdin, writes them on stdout:
  first message  (panel_uv_path, Vw, Fs, fn, depth_max, min_faces)
  then           (task_id, "piece", item)        ->  (task_id, "ok", panel_uv._process(item, ...))
                 (task_id, "xatlas", args)       ->  (task_id, "ok", panel_uv._xatlas_try(*args))   ;  end of input = quit
It loads panel_uv.py by file path (numpy / scipy only; no torch, no ComfyUI), so it starts in about a second.
"""
import importlib.util
import os
import pickle
import struct
import sys


def _read(f):
    h = f.read(8)
    if len(h) < 8:
        return None
    n = struct.unpack("<Q", h)[0]
    return pickle.loads(f.read(n))


def _write(f, obj):
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    f.write(struct.pack("<Q", len(b)) + b)
    f.flush()


def main():
    fin, fout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr                     # stray prints must not corrupt the pipe
    path, Vw, Fs, fn, depth_max, min_faces = _read(fin)
    spec = importlib.util.spec_from_file_location("mvtex_panel_uv_worker", path)
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    _write(fout, ("ready", os.getpid()))
    while True:
        msg = _read(fin)
        if msg is None:
            break
        tid, kind, payload = msg
        try:
            if kind == "piece":
                r = P._process(payload, Vw, Fs, fn, depth_max, min_faces)
            else:
                r = P._xatlas_try(*payload)
            _write(fout, (tid, "ok", r))
        except Exception as e:                  # reported to the parent, which runs this piece itself
            _write(fout, (tid, "error", repr(e)))


if __name__ == "__main__":
    main()
