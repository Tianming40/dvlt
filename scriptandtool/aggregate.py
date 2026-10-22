#!/usr/bin/env python3
"""Collect EVERY probe run under out/full_*/ into one place.

Produces, under out/full_summary/:
  summary.json  -- every run's *.json bundled verbatim (machine-readable)
  summary.txt   -- for every run and every metric found, the start vs final value
  summary.html  -- one scrollable page: all figures inline + the per-metric tables

Run from repo root:  python scriptandtool/aggregate.py
No deps beyond the standard library.
"""
import base64
import glob
import json
import os

ROOT = "out"
OUTDIR = os.path.join(ROOT, "full_summary")


def scalars(d):
    """flatten a dict to its scalar (number/str/bool) leaves, dotted keys."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, bool, str)):
            out[str(k)] = v
        elif isinstance(v, dict):
            for kk, vv in scalars(v).items():
                out[f"{k}.{kk}"] = vv
    return out


def summarize_json(obj):
    """Return a dict {metric: (first, last)} describing a run's json, whatever its shape.

    - top-level scalars -> (val, val)
    - a list of dicts (the per-step 'rows') -> each scalar key's first vs last value
    - a dict of such lists/scalars -> recurse with a prefix
    """
    rows = {}

    def take_list(prefix, lst):
        dicts = [x for x in lst if isinstance(x, dict)]
        if not dicts:
            return
        first, last = scalars(dicts[0]), scalars(dicts[-1])
        keys = list(dict.fromkeys(list(first) + list(last)))
        for k in keys:
            rows[f"{prefix}{k}"] = (first.get(k), last.get(k))

    if isinstance(obj, list):
        take_list("", obj)
    elif isinstance(obj, dict):
        for k, v in scalars(obj).items():
            rows[k] = (v, v)
        for k, v in obj.items():
            if isinstance(v, list):
                take_list(f"{k}.", v)
            elif isinstance(v, dict):
                for kk, vv in scalars(v).items():
                    rows[f"{k}.{kk}"] = (vv, vv)
    return rows


def fmt(v):
    if isinstance(v, float):
        return f"{v:.5g}"
    return "" if v is None else str(v)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    runs = sorted(d for d in glob.glob(os.path.join(ROOT, "full_*"))
                  if os.path.isdir(d) and os.path.basename(d) != "full_summary")
    if not runs:
        print("no out/full_*/ runs found -- run scriptandtool/run_all.sh first")
        return

    bundle, txt, html = {}, [], []
    html.append("<!doctype html><meta charset=utf-8><title>DVLT probe -- full summary</title>")
    html.append("<style>body{font:14px/1.5 system-ui,sans-serif;max-width:1100px;margin:24px auto;"
                "padding:0 16px;background:#fff;color:#111}h2{border-bottom:2px solid #333;margin-top:40px}"
                "img{max-width:100%;border:1px solid #ddd;margin:8px 0}table{border-collapse:collapse;"
                "font:12px monospace;margin:8px 0}td,th{border:1px solid #ccc;padding:2px 8px;text-align:right}"
                "td:first-child,th:first-child{text-align:left}@media(prefers-color-scheme:dark){"
                "body{background:#111;color:#eee}td,th{border-color:#444}img{border-color:#444}}</style>")
    html.append("<h1>DVLT probe &mdash; full run summary</h1>")
    html.append(f"<p>{len(runs)} runs. Anchors: K=16 baseline + GT. Each run = one thing changed.</p>")
    html.append("<p>" + " &middot; ".join(f'<a href="#{os.path.basename(r)}">{os.path.basename(r)}</a>'
                                           for r in runs) + "</p>")

    for run in runs:
        name = os.path.basename(run)
        txt.append("=" * 78)
        txt.append(name)
        txt.append("=" * 78)
        html.append(f'<h2 id="{name}">{name}</h2>')
        bundle[name] = {}

        for jp in sorted(glob.glob(os.path.join(run, "*.json"))):
            base = os.path.basename(jp)
            try:
                obj = json.load(open(jp))
            except Exception as e:
                txt.append(f"  [{base}] unreadable: {e}")
                continue
            bundle[name][base] = obj
            rows = summarize_json(obj)
            if not rows:
                continue
            txt.append(f"  --- {base} ---")
            txt.append(f"    {'metric':<34}{'first':>14}{'final':>14}")
            html.append(f"<h3>{base}</h3><table><tr><th>metric</th><th>first</th><th>final</th></tr>")
            for k, (a, b) in rows.items():
                txt.append(f"    {k:<34}{fmt(a):>14}{fmt(b):>14}")
                html.append(f"<tr><td>{k}</td><td>{fmt(a)}</td><td>{fmt(b)}</td></tr>")
            html.append("</table>")

        for pp in sorted(glob.glob(os.path.join(run, "*.png"))):
            b64 = base64.b64encode(open(pp, "rb").read()).decode()
            html.append(f'<div>{os.path.basename(pp)}<br>'
                        f'<img src="data:image/png;base64,{b64}"></div>')

    json.dump(bundle, open(os.path.join(OUTDIR, "summary.json"), "w"), indent=1)
    open(os.path.join(OUTDIR, "summary.txt"), "w").write("\n".join(txt) + "\n")
    open(os.path.join(OUTDIR, "summary.html"), "w").write("\n".join(html))
    print(f"wrote {OUTDIR}/summary.json  (all raw numbers)")
    print(f"wrote {OUTDIR}/summary.txt   (first-vs-final table per metric)")
    print(f"wrote {OUTDIR}/summary.html  (every figure inline + tables; open in a browser)")


if __name__ == "__main__":
    main()
