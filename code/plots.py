import simulation
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import os
import numpy as np
import save_result
from datetime import datetime
import pandas as pd
import seaborn as sns
import copy

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)
pd.set_option("display.width", 1000)


CODE_DIR = Path(__file__).parent
DATA_DIR = CODE_DIR.parent / "data"


def fig_1():
    hap_diversity, cultural_divergence = simulation.figure_1_Data()

    x = list(range(200))
    label = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(
        5, 2
    )

    for id, ax in enumerate(fig.get_axes()):

        ax.plot(x, hap_diversity[id])
        ax.xaxis.set_ticks([0, 50, 100, 150, 200])
        ax.yaxis.set_ticks([0, 0.5, 1])

        ax_twin = ax.twinx()
        color = "tab:green"
        ax_twin.plot(x, cultural_divergence[id], color=color, linestyle="dashed")
        ax_twin.text(
            0.90,
            0.95,
            label[id],
            transform=ax_twin.transAxes,
            fontsize=8,
            fontweight="bold",
            va="top",
        )
        if ax == ax6:
            ax_twin.set_ylabel("cultural_divergence (...)")
        ax_twin.yaxis.set_ticks([0, 0.05, 0.1])
        ax_twin.set_ylim(0, 0.1)

    ax9.set_xlabel("generation")
    ax10.set_xlabel("generation")
    ax5.set_ylabel("haplotype diversity(-)")

    fig.tight_layout()
    fig.savefig("fig_1")
    plt.show()


def fig_2():
    divers = simulation.figure_2_data()
    #print(divers)
    

    label = ["A: \u03C1 = 0.0001", "B: \u03C1 =0.001", "C: \u03C1 =0.1"]
    # op 1
    x = [i for i in range(200)]
    fig, ((ax1), (ax2), (ax3)) = plt.subplots(3, 1)

    for id, axs in enumerate(fig.get_axes()):
        axs.xaxis.set_ticks([0, 50, 100, 150, 200])
        axs.yaxis.set_ticks([0, 0.5, 1])
        axs.set_ylim(0, 1)
        axs.set_xlim(0, 200)

        axs.text(
            0.80,
            0.95,
            label[id],
            transform=axs.transAxes,
            fontsize=8,
            fontweight="bold",
            va="top",
        )

        for soc in divers[id]:
            axs.plot(x, soc)

    ax3.set_xlabel("generation")

    ax2.set_ylabel("haplotype diversity")
    fig.tight_layout()

    fig.savefig("fig_2")

    plt.show()
    
def fig_3():
    data_df= save_result.extract_results()
    data_df.drop(['m','tribes_num', 'initial_N', 'N_mean_tribe_size', 'Nm', 'initial haplotype_diversity', 'haplotype_diversity', 'initial nucleotide_diversity', 'nucleotide_diversity', 'cult_divergence', 'initial genetic_divergence', 'initial hs', 'initial ht', 'genetic_divergence', 'hs', 'ht', 'ev_freq', 'ev_mag', 'ev_effect', 'only_pos', 'assi_freq', 'assi_mag', 'assim_vs_innov'], axis=1, inplace=True)
    def severe(x):
        if x<50:
            return 0
        elif x>=50 and x<=90:
            return 1
        else:
            return 2

    data_df["haplotype_reduction"] = data_df['haplotype_reduction'].apply(severe )
    data_df["nucleotide_reduction"] = data_df['nucleotide_reduction'].apply(severe )
    grouped_hap = data_df.groupby(["K", "MU",'P'])["haplotype_reduction"].value_counts().reset_index(name="COUNT")
    grouped_nuc = data_df.groupby(["K", "MU",'P'])["nucleotide_reduction"].value_counts().reset_index(name="COUNT")

    def plot_bar_diagram(subplot, data, columns=None):
        if columns is not None:
            assert all(c in data.columns for c in columns)
            data = data.copy()
            columns_to_drop = [c for c in data.columns if c not in columns]
            data.drop(columns_to_drop, axis=1, inplace=True)
        xlabels = data.columns
        data.fillna(value=0, inplace=True)
        data = data.to_numpy()
    
        # data is a 2D array of shape (3, C), where C is usually 4 or 2
        assert len(data.shape) == 2
        assert data.shape[0] == 3
        columns_num = data.shape[1]
        
        barWidth = 0.25
        subplot.set_ylim(0, 10000)
        subplot.set_xticks(np.arange(columns_num) + barWidth)
        subplot.set_xticklabels(xlabels)
        subplot.tick_params(labelsize=18)

        for bar in range(3):
            bars_offsets = np.arange(columns_num) + bar * barWidth
            color = ["r", "g", "b"][bar]
            label = ["<50%", "50-90%", ">90%"][bar]
            subplot.bar(
                bars_offsets,
                data[bar], # data[bar] is a vector of size columns_num
                color=color,
                width=barWidth,
                edgecolor="grey",
                label=label,
            )

    def plot_hap_bar_diagram(subplot, k, mu, columns=None):
        df = grouped_hap
        data = df[(df["K"] == k) & (df["MU"] == mu)]
        data_2d = data.pivot(index='haplotype_reduction', columns='P', values='COUNT')
        plot_bar_diagram(subplot, data_2d, columns)
        
    def plot_nuc_bar_diagram(subplot, k, mu, columns=None):
        df = grouped_nuc
        data = df[(df["K"] == k) & (df["MU"] == mu)]
        data_2d = data.pivot(index='nucleotide_reduction', columns='P', values='COUNT')
        plot_bar_diagram(subplot, data_2d, columns)


    fig, axes = plt.subplots(5, 2, figsize=(15, 25))
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for row titles
    plt.subplots_adjust(
        wspace=0.5, hspace=0.1, bottom=0.1
    )  # Adjust space between columns and rows, and increase bottom margin

    # Create subplots    
    subdiagrams = fig.get_axes()
    plot_hap_bar_diagram(subdiagrams[0], 1000, 0.001)
    plot_nuc_bar_diagram(subdiagrams[1], 1000, 0.001)
    plot_hap_bar_diagram(subdiagrams[2], 10000, 0.001)
    plot_nuc_bar_diagram(subdiagrams[3], 10000, 0.001)
    plot_hap_bar_diagram(subdiagrams[4], 10000, 0.0001)
    plot_nuc_bar_diagram(subdiagrams[5], 10000, 0.0001)
    plot_hap_bar_diagram(subdiagrams[6], 100000, 0.0001, [1000, 10000])
    plot_nuc_bar_diagram(subdiagrams[7], 100000, 0.0001, [1000, 10000])
    plot_hap_bar_diagram(subdiagrams[8], 100000, 0.00001, [1000, 10000])
    plot_nuc_bar_diagram(subdiagrams[9], 100000, 0.00001, [1000, 10000])
    
    # Add legend to only one of the sub plots, as its identical in all plots
    subdiagrams[1].legend(fontsize=20)
    
    # Add titles, only to the top two subplots
    fig.text(
        0.25,
        0.97,
        "Haplotype diversity",
        ha="center",
        va="center",
        fontsize=24,
    )
    fig.text(
        0.75,
        0.97,
        "Nucleotide diversity",
        ha="center",
        va="center",
        fontsize=24
    )
    
    # Add fotters, only to the bottom two subplots
    subdiagrams[8].set_xlabel("Splitting Parameter, P", fontsize=24)
    subdiagrams[9].set_xlabel("Splitting Parameter, P", fontsize=24)
    
    row_titles = [
        "K=1,000\nK* \u03BC =1",
        "K=10,000\nK* \u03BC =10",
        "K=10,000\nK* \u03BC =1",
        "K=100,000\nK* \u03BC=10",
        "K=100,000\nK* \u03BC =1",
    ]
    for i, title in enumerate(row_titles):
        fig.text(
            0.5,
            0.88 - i * 0.18,
            title,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
        )
    


# def fig_3():

#     if False:
#         data = simulation.fig_3_data()
#     else:
#         # Data from pickle
#         # TODO add file name
#         with open(
#             DATA_DIR
#             / "figure_data"
#             / "fig_3"
#             / "plotting_data"
#             / "20241117_174347.pkl",
#             "rb",
#         ) as file:

#             data = pickle.load(file)
#             print(data)

#     # Bar width
#     barWidth = 0.25

#     # Set positions of bars on X axis
#     br1 = np.arange(4)
#     br2 = [x + barWidth for x in br1]
#     br3 = [x + barWidth for x in br2]

#     br4 = np.arange(2)
#     br5 = [x + barWidth for x in br4]
#     br6 = [x + barWidth for x in br5]

#     # Create subplots
#     fig, axes = plt.subplots(5, 2, figsize=(15, 25))

#     row_titles = [
#         "K=1,000\nK* \u03BC =1",
#         "K=10,000\nK* \u03BC =10",
#         "K=10,000\nK* \u03BC =1",
#         "K=100,000\nK* \u03BC=10",
#         "K=100,000\nK* \u03BC =1",
#     ]

#     for id, ax in enumerate(fig.get_axes()):
#         gen = id % 2
#         combo_i = id // 2  # Determine which dataset to use

#         ax.set_ylim(0, 2000)

#         if id < 6:  # For the first three rows (6 subplots)
#             ax.set_xticks([r + barWidth for r in range(len(br1))])
#             ax.set_xticklabels(["1", "10", "100", "1,000"])

#             # Plot bars
#             ax.bar(
#                 br1,
#                 data[combo_i][gen][0],
#                 color="r",
#                 width=barWidth,
#                 edgecolor="grey",
#                 label="small",
#             )
#             ax.bar(
#                 br2,
#                 data[combo_i][gen][1],
#                 color="g",
#                 width=barWidth,
#                 edgecolor="grey",
#                 label="medium",
#             )
#             ax.bar(
#                 br3,
#                 data[combo_i][gen][2],
#                 color="b",
#                 width=barWidth,
#                 edgecolor="grey",
#                 label="big",
#             )
#         else:  # For the last two rows (4 subplots)
#             ax.set_xticks([r + barWidth for r in range(len(br4))])
#             ax.set_xticklabels(["1,000", "10,000"])

#             # Plot bars
#             ax.bar(
#                 br4,
#                 data[combo_i][gen][0],
#                 color="r",
#                 width=barWidth,
#                 edgecolor="grey",
#                 label="small",
#             )
#             ax.bar(
#                 br5,
#                 data[combo_i][gen][1],
#                 color="g",
#                 width=barWidth,
#                 edgecolor="grey",
#                 label="medium",
#             )
#             ax.bar(
#                 br6,
#                 data[combo_i][gen][2],
#                 color="b",
#                 width=barWidth,
#                 edgecolor="grey",
#                 label="big",
#             )

#         ax.set_xlabel("Splitting Parameter, P", fontweight="bold", fontsize=12)
#         ax.legend()

#     # Adding titles for the columns
#     fig.text(
#         0.25,
#         0.97,
#         "Haplotype diversity",
#         ha="center",
#         va="center",
#         fontsize=16,
#         fontweight="bold",
#     )
#     fig.text(
#         0.75,
#         0.97,
#         "Nucleotide diversity",
#         ha="center",
#         va="center",
#         fontsize=16,
#         fontweight="bold",
#     )

#     # Adding titles for the rows
#     for i, title in enumerate(row_titles):
#         fig.text(
#             0.5,
#             0.88 - i * 0.18,
#             title,
#             ha="center",
#             va="center",
#             fontsize=16,
#             fontweight="bold",
#         )

#     plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust layout to make room for row titles
#     plt.subplots_adjust(
#         wspace=0.5, hspace=0.5, bottom=0.1
#     )  # Adjust space between columns and rows, and increase bottom margin

#     fig.savefig("fig_3")
#     plt.show()


def fig_4():

    df_cult = save_result.extract_results()

    subset = df_cult[["initial_N", "N_mean_tribe_size", "haplotype_reduction"]]
    df_data = [tuple(x) for x in subset.to_numpy()]

    small = []
    severe = []
    for x, y, reduc in df_data:
        if reduc < 10:
            small.append((x, y))
        elif reduc > 90:
            severe.append((x, y))

    # plot:
    # Initialize layout
    fig, ax = plt.subplots(figsize=(9, 6))

    # Add scatter plot
    severe_xs = [x for x, y in severe]
    severe_ys = [y for x, y in severe]
    small_xs = [x for x, y in small]
    small_ys = [y for x, y in small]
    ax.scatter(severe_xs, severe_ys, s=3, alpha=0.7, marker="+")
    ax.scatter(small_xs, small_ys, marker=".", s=3, alpha=0.7)

    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.axis((10, 100000, 10, 100000))

    ax.text(
        400,
        5000,
        "+ >90% diversity reduction\n . <10% diversity reduction",
        va="baseline",
        ha="right",
        multialignment="left",
        bbox=dict(fc="none"),
    )
    plt.xlabel("Mean tribe size before test")
    plt.ylabel("Mean tribe size after test")

    fig.savefig("fig_4")
    plt.show()


def table_2():
    data_list = simulation.data_table_2()

    # plot table

    column_labels = ["< 50%", "50%-90%", "> 90%", "Extinction"]
    row_labels = [
        "control",
        "v = 0.05",
        "v = 0.10",
        "v = 0.20",
        "v = 0.40",
        "δ = 0.0125",
        "δ = 0.0250",
        "δ = 0.0500",
        "δ = 0.1000",
    ]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Hide the axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    table = ax.table(
        cellText=data_list,
        colLabels=column_labels,
        rowLabels=row_labels,
        loc="center",
        cellLoc="center",
    )

    # Adjust font size for the entire table
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Add main title
    plt.title("Reduction in haplotype diversity", fontsize=12, fontweight="normal")

    # Adjust layout to make room for the title
    plt.subplots_adjust(top=0.85)

    # Display the table
    fig.savefig("table_2")
    plt.show()


def fig_5():
    full_data = save_result.extract_results()
    full_data["initial_Nm"] = full_data["initial_N"] * full_data["m"]
    full_data["ev_effect"] = full_data["ev_freq"] * full_data["ev_mag"]

    label_letter = ["A", "B", "C", "D", "E"]

    fig, axes = plt.subplots(5, 2, figsize=(15, 25))
    row_titles = [
        "N * m",
        "innovation rate",
        "assimilation effect",
        "cultural_divergence",
        "genetic_divergence",
    ]

    for id, ax in enumerate(fig.get_axes()):
        results, x_limits, y_lim, x_border, row_title = save_result.fig_5_rows(
            full_data, id
        )
        
        # Set positions of bars on X axis
        barWidth = 0.05
        br1 = x_limits
        # br2 = [x + barWidth for x in br1]
        # br3 = [x + barWidth for x in br2]
        # plotting row

        ax.set_xticks([r + barWidth for r in range(len(br1))])
        ax.set_xscale("log")
        if id % 2 == 0:
            ax.text(
                0.05,
                0.95,
                label_letter[id // 2],
                transform=ax.transAxes,
                fontsize=8,
                fontweight="bold",
                va="top",
            )
        for i in range(3):
            color = ["r", "g", "b"][i]
            label = ["small", "moderate", "severe"][i]

            ax.bar(
                [0] + br1[:-1],
                results[i],
                color=color,
                width=np.diff([0] + br1),
                edgecolor="grey",
                label=label,
                align="edge",
            )

        ax.set_ylim(y_lim)
        ax.set_xlim(x_border)
        ax.set_xlabel(row_title, fontweight="bold", fontsize=12)

        handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center")
    # Adding titles for the columns
    fig.text(
        0.25,
        0.97,
        "Haplotype diversity",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.75,
        0.97,
        "Nucleotide diversity",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    # Adding titles for the rows
    for i, title in enumerate(row_titles):
        fig.text(
            0.5,
            0.88 - i * 0.18,
            title,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.subplots_adjust(wspace=0.5, hspace=0.5, bottom=0.1)

    fig.savefig("fig_5")
    plt.show()


def fig_6():
    df_cult = save_result.extract_results()
    subset = df_cult[
        ["initial genetic_divergence", "genetic_divergence", "nucleotide_reduction"]
    ]
    df_data = [tuple(x) for x in subset.to_numpy()]

    small = []
    severe = []

    for x, y, reduc in df_data:
        if reduc < 10:
            small.append((x, y))
        elif reduc > 90:
            severe.append((x, y))

    # plot:

    fig, (ax1, ax2) = plt.subplots(1, 2)

    severe_xs = [x for x, y in severe]
    severe_ys = [y for x, y in severe]
    small_xs = [x for x, y in small]
    small_ys = [y for x, y in small]
    plt.axis((0, 1, 0, 1))

    # small reduction
    ax1.scatter(small_xs, small_ys, s=1)
    # big reduction
    ax2.scatter(severe_xs, severe_ys, s=1)
    x = np.linspace(0, 1)
    ax2.plot(x, x)

    ax1.set_title("<10% diversity reduction")
    ax2.set_title(">90% diversity reduction")

    ax1.set_xlabel("GST before test")
    ax1.set_ylabel("GST after test")
    ax2.set_xlabel("GST before test")
    ax2.set_ylabel("GST after test")

    fig.savefig("fig_6")
    plt.show()


def print_all():
    fig_1()
    fig_2()
    fig_3()
    fig_4()
    fig_5()
    fig_6()
    table_2()


if __name__ == "__main__":

    print_all()