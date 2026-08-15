"""Generate benchmark bar-chart SVGs for the README from Qwen3.8-27B's
official model-card numbers (https://huggingface.co/Qwen/Qwen3.8-27B).

No dependencies — emits self-contained SVG (light card bg so it stays legible
on GitHub light or dark mode). Run: python make_benchmark_charts.py
"""

SERIES = [
    ("Qwen3.8-27B", "#4C6FFF"),
    ("Qwen3.6-27B", "#9AA7BD"),
    ("Opus4.6 Max", "#F0883E"),
]

TEXT = [
    ("SWE-bench Pro", [61.7, 53.5, 53.4]),
    ("QwenSWEBench", [79.0, 49.3, 63.8]),
    ("LiveCodeBench v6", [90.3, 83.9, 88.8]),
    ("IFBench", [79.5, 69.1, 62.5]),
    ("GPQA Diamond", [89.2, 87.8, 91.3]),
    ("Terminal Bench 2.1", [73.0, 63.4, 78.2]),
]

VL = [
    ("OSWorld-Verified", [84.3, 63.9, 72.7]),
    ("AndroidWorld", [81.9, 70.3, 62.0]),
    ("MathVision", [90.0, 85.1, 65.5]),
    ("CharXiv (RQ)", [83.7, 78.4, 66.0]),
    ("RealWorldQA", [85.9, 84.1, 73.9]),
    ("OmniDocBench 1.5", [91.1, 89.4, 86.6]),
]

L, R, W = 200, 44, 780
SCALE = (W - L - R) / 100.0  # px per point, 0-100 axis
BAR_H, BAR_GAP, GROUP_PAD, Y0 = 13, 3, 22, 96
GROUP_PITCH = len(SERIES) * BAR_H + (len(SERIES) - 1) * BAR_GAP + GROUP_PAD


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def chart(title, rows):
    h = Y0 + len(rows) * GROUP_PITCH + 30
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" '
        f'viewBox="0 0 {W} {h}" font-family="system-ui,Segoe UI,Arial,sans-serif">',
        f'<rect x="0" y="0" width="{W}" height="{h}" rx="12" fill="#ffffff" stroke="#E5E7EB"/>',
        f'<text x="24" y="34" font-size="19" font-weight="700" fill="#111827">{esc(title)}</text>',
    ]
    # legend
    lx = 24
    for name, color in SERIES:
        p.append(f'<rect x="{lx}" y="52" width="13" height="13" rx="3" fill="{color}"/>')
        p.append(f'<text x="{lx + 19}" y="63" font-size="12" fill="#374151">{esc(name)}</text>')
        lx += 40 + len(name) * 7.2
    # gridlines + axis labels
    for v in (0, 25, 50, 75, 100):
        x = L + v * SCALE
        p.append(f'<line x1="{x:.1f}" y1="{Y0 - 8}" x2="{x:.1f}" y2="{h - 24}" stroke="#EEF0F3"/>')
        p.append(f'<text x="{x:.1f}" y="{h - 8}" font-size="11" fill="#9AA7BD" text-anchor="middle">{v}</text>')
    # groups
    for g, (label, vals) in enumerate(rows):
        top = Y0 + g * GROUP_PITCH
        cy = top + (len(SERIES) * BAR_H + (len(SERIES) - 1) * BAR_GAP) / 2 + 4
        p.append(f'<text x="{L - 12}" y="{cy:.1f}" font-size="13" fill="#111827" text-anchor="end">{esc(label)}</text>')
        for b, v in enumerate(vals):
            by = top + b * (BAR_H + BAR_GAP)
            bw = v * SCALE
            p.append(f'<rect x="{L}" y="{by}" width="{bw:.1f}" height="{BAR_H}" rx="2.5" fill="{SERIES[b][1]}"/>')
            p.append(f'<text x="{L + bw + 5:.1f}" y="{by + 10.5}" font-size="11" fill="#374151">{v:g}</text>')
    p.append("</svg>")
    return "\n".join(p)


def main():
    for fname, title, rows in [
        ("benchmarks_text.svg", "Text & Agent benchmarks (higher is better)", TEXT),
        ("benchmarks_vl.svg", "Vision-language benchmarks (higher is better)", VL),
    ]:
        with open(fname, "w", encoding="utf-8") as f:
            f.write(chart(title, rows))
        print(f"wrote {fname}")
    # self-check: bar never overflows the plot area
    assert 100 * SCALE + L <= W - R + 1, "bar exceeds plot width"


if __name__ == "__main__":
    main()
