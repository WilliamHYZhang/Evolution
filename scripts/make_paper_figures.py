import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

RUNS = {
    "Static ANLI": ROOT / "outputs" / "static_anli_5825418" / "metrics.csv",
    "Structured Pair": ROOT / "outputs" / "pair_anli_wsc_5825478" / "metrics.csv",
    "ANLI/MATH Pair": ROOT / "outputs" / "pair_anli_math_11817776" / "metrics.csv",
    "Structured Dynamic": ROOT / "outputs" / "run_5825378" / "metrics.csv",
    "Unstructured Dynamic": ROOT / "outputs" / "run_11817778" / "metrics.csv",
}


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def series(rows: list[dict[str, str]], task: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for row in rows:
        if row["eval_task"] == task:
            xs.append(int(row["global_step"]))
            ys.append(float(row["value"]))
    return xs, ys


def final_scores(rows: list[dict[str, str]]) -> dict[str, float]:
    finals = {}
    max_step = max(int(row["global_step"]) for row in rows)
    for row in rows:
        if int(row["global_step"]) == max_step:
            finals[row["eval_task"]] = float(row["value"])
    return finals


def plot_regimes() -> None:
    regimes = [
        ("Static ANLI", ["ANLI"] * 9),
        ("ANLI/WSC Pair", ["ANLI", "WSC"] * 4 + ["ANLI"]),
        ("ANLI/MATH Pair", ["ANLI", "MATH"] * 4 + ["ANLI"]),
        ("Structured Dynamic", ["ANLI", "WSC", "MATH"] * 3),
        ("Unstructured Dynamic", ["MATH", "ANLI", "WSC"] * 3),
    ]
    styles = {"ANLI": "anliPhase", "WSC": "wscPhase", "MATH": "mathPhase"}
    lines = [
        r"\begin{tikzpicture}[x=0.47cm,y=0.55cm,",
        r"anliPhase/.style={draw=white, fill=blue!65, text=white, minimum width=0.45cm, minimum height=0.38cm, font=\scriptsize},",
        r"wscPhase/.style={draw=white, fill=orange!85!black, text=white, minimum width=0.45cm, minimum height=0.38cm, font=\scriptsize},",
        r"mathPhase/.style={draw=white, fill=green!55!black, text=white, minimum width=0.45cm, minimum height=0.38cm, font=\scriptsize}]",
    ]
    for y, (name, phases) in enumerate(regimes):
        lines.append(rf"\node[anchor=east,font=\scriptsize] at (-0.35, {-y}) {{{name}}};")
        for x, phase in enumerate(phases):
            lines.append(rf"\node[{styles[phase]}] at ({x}, {-y}) {{{phase}}};")
    for x in range(9):
        lines.append(rf"\node[font=\scriptsize] at ({x}, 0.7) {{{x + 1}}};")
    lines.append(r"\node[font=\scriptsize] at (4, 1.15) {Training phase};")
    lines.append(r"\end{tikzpicture}")
    (OUT / "regime_sequences.tex").write_text("\n".join(lines) + "\n")


def plot_trajectories() -> None:
    tasks = [("anli", "ANLI", "blue"), ("wsc", "WSC", "orange!85!black"), ("math", "MATH", "green!55!black")]
    panels = ["Structured Dynamic", "Unstructured Dynamic"]
    lines = [
        r"\begin{tikzpicture}[x=0.0021cm,y=4.2cm]",
        r"\draw[->] (0,0) -- (1900,0) node[right,font=\scriptsize] {step};",
        r"\draw[->] (0,0) -- (0,0.72) node[above,font=\scriptsize] {accuracy};",
    ]
    for panel_idx, run_name in enumerate(panels):
        x_shift = panel_idx * 2300
        lines.append(rf"\begin{{scope}}[xshift={x_shift * 0.0021}cm]")
        lines.append(rf"\node[font=\scriptsize] at (900,0.78) {{{run_name}}};")
        for y in [0.2, 0.4, 0.6]:
            lines.append(rf"\draw[gray!25] (0,{y}) -- (1800,{y});")
            if panel_idx == 0:
                lines.append(rf"\node[anchor=east,font=\tiny] at (-40,{y}) {{{y:.1f}}};")
        rows = read_metrics(RUNS[run_name])
        for task, label, color in tasks:
            xs, ys = series(rows, task)
            coords = " -- ".join(f"({x},{y:.4f})" for x, y in zip(xs, ys))
            lines.append(rf"\draw[{color}, thick] {coords};")
            for x, y in zip(xs, ys):
                lines.append(rf"\fill[{color}] ({x},{y:.4f}) circle[radius=1.8pt];")
            lines.append(rf"\node[{color},font=\tiny,anchor=west] at (1820,{ys[-1]:.4f}) {{{label}}};")
        for x in [0, 600, 1200, 1800]:
            lines.append(rf"\draw[gray!45] ({x},0) -- ({x},-0.015) node[below,font=\tiny] {{{x}}};")
        lines.append(r"\end{scope}")
    lines.append(r"\end{tikzpicture}")
    (OUT / "dynamic_trajectories.tex").write_text("\n".join(lines) + "\n")


def plot_final_scores() -> None:
    names = list(RUNS.keys())
    data = {name: final_scores(read_metrics(path)) for name, path in RUNS.items()}
    tasks = [("anli", "ANLI"), ("wsc", "WSC"), ("math", "MATH")]
    colors = {"anli": "blue!65", "wsc": "orange!85!black", "math": "green!55!black"}
    lines = [
        r"\begin{tikzpicture}[x=1.25cm,y=5.2cm]",
        r"\draw[->] (-0.7,0) -- (5.3,0);",
        r"\draw[->] (-0.7,0) -- (-0.7,0.72) node[above,font=\scriptsize] {final accuracy};",
    ]
    for y in [0.2, 0.4, 0.6]:
        lines.append(rf"\draw[gray!25] (-0.7,{y}) -- (5.1,{y});")
        lines.append(rf"\node[anchor=east,font=\tiny] at (-0.75,{y}) {{{y:.1f}}};")
    for i, name in enumerate(names):
        short_name = name.replace("Structured ", "Struct. ").replace("Unstructured ", "Unstruct. ").replace("Static ", "Static ")
        lines.append(rf"\node[rotate=25,anchor=east,font=\tiny] at ({i},-0.03) {{{short_name}}};")
        for j, (task, label) in enumerate(tasks):
            x = i + (j - 1) * 0.22
            val = data[name][task]
            lines.append(rf"\draw[fill={colors[task]}, draw=white] ({x - 0.10},0) rectangle ({x + 0.10},{val:.4f});")
    for j, (task, label) in enumerate(tasks):
        lines.append(rf"\draw[fill={colors[task]}, draw=white] (3.7,{0.70 - j * 0.06}) rectangle (3.9,{0.73 - j * 0.06});")
        lines.append(rf"\node[anchor=west,font=\tiny] at (3.95,{0.715 - j * 0.06}) {{{label}}};")
    lines.append(r"\end{tikzpicture}")
    (OUT / "final_accuracy_by_regime.tex").write_text("\n".join(lines) + "\n")

    with (ROOT / "paper" / "results_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["regime", "anli", "wsc", "math"])
        for name in names:
            writer.writerow([name, data[name]["anli"], data[name]["wsc"], data[name]["math"]])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_regimes()
    plot_trajectories()
    plot_final_scores()


if __name__ == "__main__":
    main()
