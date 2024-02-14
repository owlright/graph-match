import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import argparse
import os

# * default values
figure_dir = f"C:/Users/SJH/Documents/thesis/figures/"
# reading data from file

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format',
                    help="the type of output picture",
                    default="tif",
                    type=str,
                    )
parser.add_argument('-t', '--topo',
                    help="the input file name",
                    default="bcube",
                    type=str)

parser.add_argument('-d', '--directory',
                    help="the input file name",
                    default=os.getcwd(),
                    type=str)

# parser.add_argument('-o', '--output',
#                     help="picture you want to output",
#                     nargs='+',
#                     default=['opt', 'gcat'],
#                     type=str)

args = parser.parse_args()
figure_dir = args.directory
pic_type = args.format
input_file = args.topo + '-bruteforce-result.txt'
# plot_which = args.output
# print(plot_which)

# reading data from file
resultFile = open(input_file)
number_of_experiments = int(resultFile.readline().strip())
topo = resultFile.readline().strip()
assert topo == args.topo

bruteforcecosts = {}
gcatcosts = {}
originalcosts = {}
bruterforceused = {}
gcatused = {}
bruteforce_compressratio = {}
gcat_compressratio = {}
bruteforce_eff= {}
gcat_eff = {}

for topo in topos:
  bruteforcecosts[topo] = []; bruterforceused[topo] = []
  gcatcosts[topo] = []; gcatused[topo] = []
  originalcosts[topo] = []
  bruteforce_compressratio[topo] = []
  gcat_compressratio[topo] = []
  bruteforce_eff[topo] = []
  gcat_eff[topo] = []

for topo in topos:
  line = resultFile.readline().strip()
  assert line == topo
  for _ in range(number_of_experiments):
    orginal_cost = int(resultFile.readline().strip())
    originalcosts[topo].append(orginal_cost)
    line = resultFile.readline()
    bcost = []
    bused = []
    gcost = []
    gused = []
    bcomp = []
    gcomp = []
    beff = []
    geff = []
    while line != "\n":
      data = line.split()
      bcost.append(int(data[1]))
      gcost.append(int(data[2]))
      bused.append(int(data[3]))
      gused.append(int(data[4]))
      bcomp.append(1 - int(data[1])/orginal_cost)
      gcomp.append(1 - int(data[2])/orginal_cost)
      beff.append(bcomp[-1] / int(data[3]))
      geff.append(gcomp[-1]/ int(data[4]))
      line = resultFile.readline()
    bcost.pop()
    bused.pop()
    gcost.pop()
    gused.pop()
    bcomp.pop()
    gcomp.pop()
    beff.pop()
    geff.pop()
    bruteforcecosts[topo].append(bcost)
    gcatcosts[topo].append(gcost)
    bruterforceused[topo].append(bused)
    gcatused[topo].append(gused)
    bruteforce_compressratio[topo].append(bcomp)
    gcat_compressratio[topo].append(gcomp)
    bruteforce_eff[topo].append(beff)
    gcat_eff[topo].append(geff)
resultFile.close()

algos = ["OPT", "GCAT"]
# print(algos)
for topo in topos:
  print(topo)
  print("compress", "effiency")
  y1 = []
  y2 = []
  for algo in algos:
    if algo == "OPT":
      comp = bruteforce_compressratio[topo]
      eff = bruteforce_eff[topo]
    elif algo == "GCAT":
      comp = gcat_compressratio[topo]
      eff = gcat_eff[topo]
    # oc = originalcosts[topo]
    exp1 = [sum(tmp)/len(tmp) for tmp in comp]
    exp2 = [sum(tmp)/len(tmp) for tmp in eff]
    y1.append(sum(exp1)/len(exp1))
    y2.append(sum(exp2)/len(exp2))
  print(y1)
  print(y2)


# math_font = "stix"
# chinese_font = mpl.font_manager.FontProperties(fname="fonts/simsun.ttc")
# chinese_font._math_fontfamily = math_font
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["mathtext.fontset"] = math_font

# colors = ['k',  'red']
# hatches = ['x', '..']



# print(topos)
# for algo in algos:
#   y = []
#   for topo in topos:
#     if algo == "OPT":
#       used = bruterforceused[topo]
#     elif algo == "GCAT":
#       used = gcatused[topo]
#     # oc = originalcosts[topo]
#     exp = []
#     # index = algos.index(algo)
#     for k in range(number_of_experiments):
#       # compressratio = []
#       m = len(bruterforceused[k])
#       for i in range(m):
#         compressratio.append(bruterforceused[k][i]/)
#       exp.append(1-sum(compressratio)/m)
#     y.append(sum(exp)/len(exp))
#   print(algo, y)


# bar_width = 0.2
# x = np.arange(1, 1+len(algos))
# fig, ax = plt.subplots()
# ax.set_ylabel(r"代价减少率$\theta$", fontproperties=chinese_font, fontsize=14)
# offset = 0


  # p = ax.bar(x+bar_width*offset, y,bar_width, fill=False, edgecolor=colors[index], hatch=hatches[index],  align='edge', label=algo)
  # ax.bar_label(p, fmt='%.2f', size=10, padding=1, label_type='edge')
  # offset += 1

# for algo in algos:
#   y = []
#   for topo in topos:
#     if algo == "OPT":
#       costs = bruteforcecosts[topo]
#     elif algo == "GCAT":
#       costs = gcatcosts[topo]
#     oc = originalcosts[topo]
#     exp = []
#     index = algos.index(algo)
#     for k in range(number_of_experiments):
#       compressratio = []
#       m = len(costs[k])
#       for i in range(m):
#         compressratio.append(costs[k][i]/oc[k])
#       exp.append(1-sum(compressratio)/m)
#     y.append(sum(exp)/len(exp))
#   p = ax.bar(x+bar_width*offset, y,bar_width, fill=False, edgecolor=colors[index], hatch=hatches[index],  align='edge', label=algo)
#   ax.bar_label(p, fmt='%.2f', size=10, padding=1, label_type='edge')
#   offset += 1

# ax.set_xticks(x+bar_width, topos)
# ax.set_ylim(0, 0.5)
# ax.set_xlim(0,4)
# ax.tick_params(axis='y',direction="in")
# ax.tick_params(labelsize=12) # change both ticks of x and y
# leg = ax.legend(fontsize=12, loc='upper right')
# # leg.get_frame().set_linewidth(1)
# leg.get_frame().set_edgecolor('k')
# leg.get_frame().set_boxstyle('square', pad=0)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.savefig(os.path.join(figure_dir, f"compare_with_opt."+pic_type), dpi=600, bbox_inches='tight')
# plt.show()
