import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib import font_manager
import numpy as np
import argparse

# import getopt
import sys
import os

# * default values
# figure_dir = f"C:/Users/SJH/Documents/thesis/figures/"
# reading data from file

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--format",
    help="the type of output picture",
    default="tif",
    type=str,
)

parser.add_argument("-i", "--input", help="input file name", default=None, type=str)

parser.add_argument("-d", "--directory", help="output dictory", default=".", type=str)

parser.add_argument("-t", "--topo", help="topo name", default="fattree", type=str)

parser.add_argument(
    "-m",
    "--metric",
    help="which metric you want see",
    nargs="+",
    default=["compression", "resource", "eff"],
    type=str,
)

args = parser.parse_args()
figure_dir = args.directory
pic_type = args.format
input_file = args.input
plot_which = args.topo
metrics = args.metric
print(plot_which)
if input_file is not None:
    result_file = open(input_file)
else:
    input_file = "exp1-" + plot_which + "-result.txt"
    result_file = open(input_file)

line = result_file.readline()
algnames = line.strip("[]\n").split(", ")
algnames = [t.strip("'") for t in algnames]
# base_index = 2*(len(tablenames))
print(f"algo names: {algnames}")

# line = result_file.readline()
# algos = line.strip("[]\n").split(", ")
# algos = [t.strip("'") for t in algos]
# print(f'algo names: {algos}')

line = result_file.readline().strip()
assert line == plot_which
print(f"reading data from {plot_which}...")
# topos = line.strip("[]\n").split(", ")
# topos = [t.strip("'") for t in topos]
# print(f'topo names: {topos}')

line = result_file.readline()
resource = line.strip("[]\n").split(", ")
resource = [float(t) for t in resource]
print(f"resource: {resource}")

line = result_file.readline()
capacity = line.strip("[]\n").split(", ")
capacity = [float(t) for t in capacity]
print(f"capacity: {capacity}")

line = result_file.readline()
number_of_experiments = int(line)
print(f"number of experiments: {number_of_experiments}")

tree_index = 2
S_index = 3
base_index = 10
compression = {}
resourceuse = {}
resourceactual = {}
cr = {}
# for topo in topos:
#   compression[topo] = {}
#   resourceuse[topo] = {}
#   resourceactual[topo] = {}
#   cr[topo] = {}
# for topo in topos:
# line = result_file.readline().strip()
origin_traffic_index = 5  # ! middle column under shortest
for algo in algnames:
    compression[algo] = np.empty([len(resource), len(capacity), number_of_experiments])

    resourceuse[algo] = np.empty([len(resource), len(capacity), number_of_experiments])

    resourceactual[algo] = np.empty(
        [len(resource), len(capacity), number_of_experiments]
    )

for expno in range(number_of_experiments):
    for res in range(len(resource)):
        for cap in range(len(capacity)):
            line = result_file.readline().strip("()\n").split(", ")
            origin_traffic = int(line[origin_traffic_index])  # algo none
            number_of_trees = int(line[tree_index])
            number_of_sources = int(line[S_index])
            for index, algo in enumerate(algnames):
                traffic = int(line[base_index + 2 * index])
                used = int(line[base_index + 2 * index + 1])
                compression[algo][res, cap, expno] = round(traffic / origin_traffic, 3)
                resourceuse[algo][res, cap, expno] = round(
                    used / int(number_of_trees * number_of_sources * capacity[cap]), 3
                )
                resourceactual[algo][res, cap, expno] = used
                # cr[topo][algo][expno, rich] = (1-compression[topo][algo][expno, rich])*100/(used)
print(f"Load finished!")

result_file.close()
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

markers = ["^", "o", "x", "+"]
markersizes = [7, 7, 7, 7]
colors = ["k", "blue", "red", "c"]
hatches = ["//", r"\\", "..", "x"]
tick_label = [
    r"$①\alpha \leq 0.5;\beta \leq 0.5$",
    r"$②\alpha \leq 0.5;\beta > 0.5$",
    r"$③\alpha > 0.5;\beta \leq 0.5$",
    r"$④\alpha > 0.5;\beta > 0.5$",
]

assert compression[algo].shape[0] == compression[algo].shape[1]
datalen = compression[algo].shape[0]
datamid = int(compression[algo].shape[0] / 2)
for metric in metrics:
    if metric == "compression":
        fig, ax = plt.subplots()
        ax.set_ylabel(r"代价减少率$\theta$", fontproperties=chinese_font, fontsize=14)

        x = np.arange(4)
        bar_width = 0.2
        count = 0
        # colors = []
        for algo in algnames:
            y = []
            y.append(1 - np.mean(compression[algo][0:datamid, 0:datamid]))
            y.append(1 - np.mean(compression[algo][0:datamid, datamid:datalen]))

            y.append(1 - np.mean(compression[algo][datamid:datalen, 0:datamid]))
            y.append(1 - np.mean(compression[algo][datamid:datalen, datamid:datalen]))

            p = ax.bar(
                x + bar_width * count,
                y,
                bar_width,
                fill=False,
                edgecolor=colors[count],
                hatch=hatches[count],
                label=algo,
            )
            ax.bar_label(p, fmt="%.2f", size=10, padding=1, label_type="edge")
            count += 1
        ax.set_xticks(x + bar_width, tick_label)
        ax.tick_params(axis="y", direction="in")
        ax.tick_params(labelsize=12)  # change both ticks of x and y
        ax.set_ylim(0, 0.8)
        leg = ax.legend(fontsize=12, loc="upper left", ncol=2)
        # leg.get_frame().set_linewidth(1)
        leg.get_frame().set_edgecolor("k")
        leg.get_frame().set_boxstyle("square", pad=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.savefig(
            os.path.join(figure_dir, f"{plot_which}-compression." + pic_type),
            dpi=600,
            bbox_inches="tight",
        )

    elif metric == "eff":
        fig, ax = plt.subplots()
        # ax.set_xlabel(r"不同负载场景", fontproperties="Simsun", fontsize=14)
        ax.set_ylabel(r"资源效率$\eta/\%$", fontproperties=chinese_font, fontsize=14)
        x = np.arange(4)
        bar_width = 0.2
        count = 0
        for algo in algnames:
            y = []
            y.append(
                100
                * (1 - np.mean(compression[algo][0:datamid, 0:datamid]))
                / np.mean(resourceactual[algo][0:datamid, 0:datamid])
            )
            y.append(
                100
                * (1 - np.mean(compression[algo][0:datamid, datamid:datalen]))
                / np.mean(resourceactual[algo][0:datamid, datamid:datalen])
            )
            # y.append(np.mean(compression[topo][algo][5, 5, :]))
            y.append(
                100
                * (1 - np.mean(compression[algo][datamid:datalen, 0:datamid]))
                / np.mean(resourceactual[algo][datamid:datalen, 0:datamid])
            )
            y.append(
                100
                * (1 - np.mean(compression[algo][datamid:datalen, datamid:datalen]))
                / np.mean(resourceactual[algo][datamid:datalen, datamid:datalen])
            )
            # y.append(np.mean(compression[topo][algo][-1, -1, :]))
            p = ax.bar(
                x + bar_width * count,
                y,
                bar_width,
                fill=False,
                edgecolor=colors[count],
                hatch=hatches[count],
                label=algo,
            )
            ax.bar_label(p, fmt="%.2f", padding=1, label_type="edge")
            count += 1
        ax.set_xticks(x + bar_width, tick_label)
        ax.tick_params(axis="y", direction="in")
        ax.tick_params(labelsize=12)
        # ax.set_ylim(0, 0.9)
        leg = ax.legend(fontsize=12, ncol=2, loc="upper right")
        # leg.get_frame().set_linewidth(1)
        leg.get_frame().set_edgecolor("k")
        leg.get_frame().set_boxstyle("square", pad=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # plt.show()

        plt.savefig(
            os.path.join(figure_dir, f"{plot_which}-efficiency." + pic_type),
            dpi=600,
            bbox_inches="tight",
        )
    # plt.savefig(f"{topo}-efficiency.pdf", dpi=300, bbox_inches='tight')
