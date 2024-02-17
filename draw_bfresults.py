import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import numpy as np

topos = ["Fattree", "Torus", "BCube", "ER-random"]
x = np.arange(len(topos))

markersizes = [7, 7, 7, 7]


bar_width = 0.2
last_position = 0
space = 2 * bar_width
fig, ax = plt.subplots()
ax.set_ylim(0, 1.0)
# ! for chinese text use
math_font = "stix"
chinese_font = mpl.font_manager.FontProperties(fname="fonts/simsun.ttc")
chinese_font._math_fontfamily = math_font
# draw pictures
blank = 0.1
# matplotlib.use('pgf') # 修改绘图后端
# default use
enfontfile = "fonts/times.ttf"
font_manager.fontManager.addfont(enfontfile)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = math_font
plt.rcParams["font.sans-serif"] = [
    "Times New Roman",
    "DejaVu Sans",
]


for topo in ["fattree", "bcube", "torus", "random"]:  # , "torus", "bcube", "random"
    hatches = ["//", r"\\", ".."]
    colors = ["k", "blue", "red"]
    if topo == "fattree":
        hatches.append("xx")
        colors.append("c")
    elif topo == "bcube":
        hatches.append("++")
        colors.append("g")
    elif topo == "torus":
        hatches.append("*")
        colors.append("m")
    elif topo == "random":
        pass
    else:
        assert False
    file_name = "exp2-" + topo + "-result.txt"
    resultFile = open(file_name)
    number_of_experiments = int(resultFile.readline().strip())
    topo = resultFile.readline().strip()
    algnames = []
    for alg in resultFile.readline().strip("[]\n").split(", "):
        alg = alg[1:-1]
        if alg != "OPT":
            algnames.append(alg)

    theta = {alg: [] for alg in algnames}
    eta = {alg: [] for alg in algnames}
    for _ in range(number_of_experiments):
        line = resultFile.readline().strip()
        origin_cost = int(line)
        while True:
            line = resultFile.readline()
            if line == "\n":
                break
            data = line.strip("[]\n").split(", ")
            opt_cost = int(data[0])
            opt_used = int(data[1])
            for i in range(len(algnames)):
                alg_reduced = origin_cost - int(data[2 * i + 2])
                opt_reduced = origin_cost - opt_cost
                theta[algnames[i]].append(alg_reduced / opt_reduced)
    resultFile.close()
    for k, v in theta.items():
        print(k, sum(v) / len(v))
    for index, alg in enumerate(algnames):
        p = ax.bar(
            (bar_width) * index + bar_width / 2 + last_position,
            sum(theta[alg]) / len(theta[alg]),
            bar_width,
            fill=False,
            edgecolor=colors[index],
            hatch=hatches[index],
            label=alg,
        )
        ax.bar_label(p, fmt="%.2f", size=10, padding=1, label_type="edge")

    last_position = (bar_width) * index + bar_width / 2 + last_position + space
ax.set_xticks(
    [
        bar_width * 2,
        space + bar_width * 5.5,
        2 * space + bar_width * 9,
        3 * space + bar_width * 12,
    ],
    topos,
)
handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center')
unique_handles = []
unique_labels = []
visited = set()
for h, l in zip(handles, labels):
    if l not in visited:
        unique_handles.append(h)
        unique_labels.append(l)
        visited.add(l)
leg = ax.legend(
    unique_handles,
    unique_labels,
    fontsize=10,
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, 1.10),
)
leg.get_frame().set_edgecolor("k")
leg.get_frame().set_boxstyle("square", pad=0)

ax.set_ylabel(r"$\theta$/$\theta_{\rm{OPT}}$", fontproperties=chinese_font, fontsize=14)
ax.tick_params(axis="y", direction="in")
ax.tick_params(labelsize=12)  # change both ticks of x and y
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.savefig("bruteforce-theta.tif", dpi=600, bbox_inches="tight")
