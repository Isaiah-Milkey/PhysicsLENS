"""
Assemble every result report into ONE self-contained, pdflatex-compilable .tex.

No pandoc on this machine, so this carries a small Markdown -> LaTeX converter
covering what the reports use: headings, paragraphs, **bold**, *italic*,
`code`, links, pipe tables, bullet / numbered lists, fenced code, block quotes,
horizontal rules. Unicode is mapped to LaTeX (pdflatex, Overleaf's default,
fails on raw ρ/Δ/≤), and wide tables are scaled to the text width.

Output: eval_reports/ALL_RESULTS.tex

python backend/scripts/build_results_tex.py
"""
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval_reports" / "ALL_RESULTS.tex"

UNI = {"—": "---", "–": "--", "−": "$-$", "→": r"$\rightarrow$", "↔": r"$\leftrightarrow$",
       "↓": r"$\downarrow$", "✓": r"\checkmark{}", "✗": r"$\times$", "ρ": r"$\rho$",
       "Δ": r"$\Delta$", "×": r"$\times$", "·": r"\textperiodcentered{}", "§": r"\S{}",
       "≥": r"$\geq$", "≤": r"$\leq$", "…": r"\ldots{}", "±": r"$\pm$", "⚠": "(!)",
       "≈": r"$\approx$", "κ": r"$\kappa$", "θ": r"$\theta$", "“": "``", "”": "''",
       "’": "'", "‘": "`"}
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}",
           "|": r"\textbar{}"}


def esc(s):
    out = []
    for ch in s:
        if ch in SPECIAL:
            out.append(SPECIAL[ch])
        elif ch in UNI:
            out.append(UNI[ch])
        elif ord(ch) > 127:
            out.append("?")
        else:
            out.append(ch)
    return "".join(out)


_TOKS = []


def inline(s):
    """Markdown inline -> LaTeX. Code spans are protected first so their
    contents are escaped literally and never parsed as emphasis."""
    toks = _TOKS          # shared across nested calls: inner spans reuse the store

    def keep(latex):
        toks.append(latex)
        return f"\x00{len(toks)-1}\x00"
    s = re.sub(r"`([^`]+)`", lambda m: keep(r"\texttt{" + esc(m.group(1)) + "}"), s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: keep(esc(m.group(1))), s)
    s = re.sub(r"\*\*(.+?)\*\*", lambda m: keep(r"\textbf{" + inline(m.group(1)) + "}"), s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])",
               lambda m: keep(r"\emph{" + inline(m.group(1)) + "}"), s)
    s = re.sub(r"(?<![\w])_(?!\s)([^_]+?)(?<!\s)_(?![\w])",
               lambda m: keep(r"\emph{" + inline(m.group(1)) + "}"), s)
    s = esc(s)
    # placeholders may nest (bold containing italic), so expand until none remain
    while "\x00" in s:
        s = re.sub(r"\x00(\d+)\x00", lambda m: toks[int(m.group(1))], s)
    return s


def table(rows):
    rows = [r for r in rows if not re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*", r)]
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    n = max(len(r) for r in cells)
    cells = [r + [""] * (n - len(r)) for r in cells]
    spec = "l" + "c" * (n - 1)
    body = [r"\toprule", " & ".join(r"\textbf{" + inline(c) + "}" for c in cells[0]) + r" \\",
            r"\midrule"]
    for r in cells[1:]:
        body.append(" & ".join(inline(c) for c in r) + r" \\")
    body.append(r"\bottomrule")
    tab = "\\begin{tabular}{" + spec + "}\n" + "\n".join(body) + "\n\\end{tabular}"
    width = max(sum(len(c) for c in r) for r in cells)
    if n > 5 or width > 95:
        tab = r"\resizebox{\linewidth}{!}{%" + "\n" + tab + "}"
    return "\\begin{center}\\small\n" + tab + "\n\\end{center}"


def md_to_tex(md, level=0):
    """level shifts headings so each report nests under its own \\section."""
    heads = [r"\section", r"\subsection", r"\subsubsection", r"\paragraph", r"\subparagraph"]
    out, i, L = [], 0, md.split("\n")
    lst = None
    para = []

    def flush():
        nonlocal para
        if para:
            out.append(inline(" ".join(p.strip() for p in para)))
            out.append("")
            para = []

    def close_list():
        nonlocal lst
        if lst:
            out.append(r"\end{" + lst + "}")
            lst = None
    while i < len(L):
        ln = L[i]
        if ln.strip().startswith("```"):
            flush(); close_list()
            j = i + 1
            code = []
            while j < len(L) and not L[j].strip().startswith("```"):
                code.append(L[j])
                j += 1
            safe = "\n".join("".join(UNI.get(ch, ch if ord(ch) < 128 else "?") for ch in c)
                             for c in code)
            out.append("\\begin{small}\\begin{verbatim}\n" + safe + "\n\\end{verbatim}\\end{small}")
            i = j + 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            flush(); close_list()
            k = min(max(len(m.group(1)) - 2 + level, 1), len(heads) - 1)
            star = "*" if k == 0 and level == 0 else ""
            out.append(f"{heads[k]}{star}{{{inline(m.group(2))}}}")
            i += 1
            continue
        if ln.strip().startswith("|"):
            flush(); close_list()
            rows = []
            while i < len(L) and L[i].strip().startswith("|"):
                rows.append(L[i])
                i += 1
            out.append(table(rows))
            continue
        if re.fullmatch(r"\s*(-{3,}|\*{3,})\s*", ln):
            flush(); close_list()
            out.append(r"\medskip\noindent\rule{\linewidth}{0.3pt}\medskip")
            i += 1
            continue
        m = re.match(r"^\s*([-*]|\d+\.)\s+(.*)$", ln)
        if m:
            flush()
            kind = "enumerate" if m.group(1)[0].isdigit() else "itemize"
            if lst != kind:
                close_list()
                out.append(r"\begin{" + kind + "}")
                lst = kind
            item = [m.group(2)]
            i += 1
            while i < len(L) and L[i].startswith("  ") and L[i].strip() and \
                    not re.match(r"^\s*([-*]|\d+\.)\s+", L[i]):
                item.append(L[i].strip())
                i += 1
            out.append(r"\item " + inline(" ".join(item)))
            continue
        if ln.startswith(">"):
            flush(); close_list()
            q = []
            while i < len(L) and L[i].startswith(">"):
                q.append(L[i].lstrip("> "))
                i += 1
            out.append(r"\begin{quote}" + inline(" ".join(q)) + r"\end{quote}")
            continue
        if not ln.strip():
            flush(); close_list()
            i += 1
            continue
        close_list()
        para.append(ln)
        i += 1
    flush(); close_list()
    return "\n".join(out)


def verbatim_block(title, text):
    safe = "\n".join("".join(UNI.get(ch, ch if ord(ch) < 128 else "?") for ch in ln)
                     for ln in text.split("\n"))
    return (f"\\subsection{{{title}}}\n\\begin{{scriptsize}}\\begin{{verbatim}}\n"
            f"{safe}\n\\end{{verbatim}}\\end{{scriptsize}}")


def main():
    # fresh system-evaluation output (fixed-width text, kept verbatim)
    se = {}
    for scope in ("pairs", "all"):
        r = subprocess.run([sys.executable, str(ROOT / "backend/scripts/system_eval.py"),
                            "--data", "data/consol", "--scope", scope],
                           cwd=ROOT, capture_output=True, text=True)
        se[scope] = "\n".join(l for l in r.stdout.split("\n")
                              if "Warning" not in l and not l.startswith("->"))

    tabs = ROOT / "paper" / "tables"
    parts = [
        ("Part I --- Consolidated robot benchmark (paper data, Sep 24)", None),
        ("Summary report", ROOT / "eval_reports/CONSOLIDATED_REPORT.md"),
        ("Paper tables", "PAPER"),
        ("All further analyses (sections A--Z)", ROOT / "eval_reports/CONSOLIDATED_ANALYSES.md"),
        ("Full system-evaluation output", "SYSTEM"),
        ("Part II --- RobotBench study: 15 experiments (Sep 5--11)", None),
        ("Report", ROOT / "report.md"),
        ("Part III --- VideoPhy-2 study (Aug)", None),
        ("Results", ROOT / "eval_reports/results.md"),
    ]
    body = []
    for title, src in parts:
        if src is None:
            body.append(r"\part{" + esc(title).replace("---", "---") + "}")
            continue
        body.append(r"\section{" + esc(title) + "}")
        if src == "PAPER":
            body.append("Generated by \\texttt{backend/scripts/paper\\_tables.py}; identical to "
                        "the files under \\texttt{paper/tables/}.")
            for f in ("models", "stagewise", "categories", "agreement"):
                t = (tabs / f"{f}.tex").read_text()
                t = t.replace(r"\begin{table}[t]", r"\begin{table}[H]")
                body.append(t)
        elif src == "SYSTEM":
            body.append("Raw output of \\texttt{backend/scripts/system\\_eval.py}: every "
                        "judge (standard and debiased prompt) $\\times$ system (VLM only, "
                        "tool only, tool + VLM) $\\times$ metric.")
            body.append(verbatim_block("Obs/unobs pairs (119 pairs, 238 clips)", se["pairs"]))
            body.append(verbatim_block("All 439 clips", se["all"]))
        else:
            md = src.read_text()
            md = re.sub(r"\A# .*\n", "", md)           # drop the report's own H1
            body.append(md_to_tex(md, level=1))

    doc = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.9in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{caption}}
\usepackage[colorlinks=true,linkcolor=blue!50!black]{{hyperref}}
\setlength{{\parskip}}{{4pt}}
\setlength{{\parindent}}{{0pt}}
\title{{PhysicsLENS --- all evaluation results and runs}}
\author{{Som Sagar}}
\date{{Compiled {date.today():%B %d, %Y}}}
\begin{{document}}
\maketitle
\begin{{abstract}}
Every result and run from the PhysicsLENS evaluation work in one document.
Part~I is the consolidated robot benchmark used in the paper (439 annotated
clips across four generators, 119 observable/unobservable pairs, 80 real
demonstrations, 12 VLM judges; VideoPhy-2 as a held-out check). Part~II is the
earlier 15-experiment RobotBench study. Part~III is the original VideoPhy-2
study. Numbers are copied from generated reports; the producing script is named
next to each table. Where Part~I supersedes an earlier conclusion (notably the
effect of model scale, which reverses under the debiased prompt), Part~I is the
current result.
\end{{abstract}}
\tableofcontents
\newpage
{chr(10).join(body)}
\end{{document}}
"""
    OUT.write_text(doc)
    print(f"-> {OUT}  ({len(doc.splitlines())} lines)")


if __name__ == "__main__":
    main()
