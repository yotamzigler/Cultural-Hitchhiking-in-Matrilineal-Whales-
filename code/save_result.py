import csv
import numpy as np
import pandas as pd
import state
from pathlib import Path
import os

CODE_DIR = Path(__file__).parent
DATA_DIR = CODE_DIR.parent / "data"
RES_DIR = CODE_DIR.parent / "result"


## Diversity measurements########################################################################################################################################################################################################


def cv_fitness(soc: state.Society):
    all_w = [c.w for c in soc.tribes]

    mean = np.mean(all_w)
    if mean == 0.0:
        return 0
    if len(soc.tribes) == 1:
        return 0
    # ddof are levels of freedom were left 0 (instead of 1, measuring the entire population, not a sample )
    std_w = np.std(all_w, ddof=1)

    c_v = std_w / mean
    assert c_v >= 0
    return c_v


def Genetic_Divergence_GST(soc, haplotypes_n):
    ## GST is used to assess the genetic structure of populations
    ## HS is the Heterozygosity within tribes(the probability that two randomly chosen alleles are different)
    # HT is the Total Heterozygosity (regardless of the Subpopulations)

    n = soc.total_size()
    # calculate HT-
    ht = GST_helper(n, haplotypes_n)

    if ht == 0:
        return (0, 0, 0)
    # calculate HS-
    hs = sum(GST_helper(c.size(), c.n_h) * c.size() for c in soc.tribes) / n

    # computing result using GST formula.
    # Due to cases where numerical precision causes slight negative values, I added the max function to ensure GST is non-negative
    gst = max(0, (ht - hs) / ht)
    return (gst, hs, ht)


def GST_helper(n: int, haplotypes_dict: dict):
    if n <= 1:
        return 0
    sum_hs = sum((h_n / n) * ((h_n - 1) / (n - 1)) for h_n in haplotypes_dict.values())
    return 1 - sum_hs


def nucleotide_diversity(soc: state.Society, haplotypes_n):
    ## using pi estimator without the unbiased estimator  correction.
    # "haplotypes_n" is the number of individuals with each haplotype in the society.
    n = soc.total_size()
    key_list = list(haplotypes_n.keys())
    if len(key_list) == 1:
        return 0

    # comparing each h to all others
    sum = 0
    for h1 in key_list:
        for h2 in key_list[1:]:
            # pi is the number of bp differences between h1 and h2
            pi = soc.calculate_bp_dif(h1, h2)
            sum += (haplotypes_n[h1] / n) * (haplotypes_n[h2] / n) * pi

    return sum


def haplotype_diversity(soc: state.Society, haplotypes_n):
    # left side of equation 5
    # "haplotypes_n" is the number of individuals with each haplotype in the society.
    size = soc.total_size()
    if size <= 1:
        return 0

    sig_h = sum(n**2 for n in haplotypes_n.values())
    numerator = size**2 - sig_h

    return numerator / (size * (size - 1))


## calculations########################################################################################################################################################################################################


def reduction_percent_951(soc: state.Society, property):
    ## this function returns 2 for "severe reduction" 1 for "moderate" and 0 for "little" reduction
    ## it is used for table 2 calculations and Fig.3
    hap_n = soc.haplotype_count()

    if property == "haplotype":

        o_d = soc.initial_haplotype_diversity
        n_d = haplotype_diversity(soc, hap_n)

    elif property == "nucleotide":
        o_d = soc.initial_nucleotide_diversity
        n_d = nucleotide_diversity(soc, hap_n)

    fifty_percent_reduction = o_d * 0.5
    ninety_percent_reduction = o_d * 0.1

    if n_d <= ninety_percent_reduction:
        return 2
    elif n_d <= fifty_percent_reduction:
        return 1
    else:
        return 0


def reduction_percent(soc: state.Society, property):
    hap_n = soc.haplotype_count()

    if property == "haplotype":

        o_d = soc.initial_haplotype_diversity
        n_d = haplotype_diversity(soc, hap_n)

    elif property == "nucleotide":
        o_d = soc.initial_nucleotide_diversity
        n_d = nucleotide_diversity(soc, hap_n)

    amount_reduc = o_d - n_d
    reduc_percent = (amount_reduc * 100) / o_d

    return reduc_percent


## work with csv########################################################################################################################################################################################################


def create_one_csv():

    filename = RES_DIR / "full_results.csv"

    header = (
        "K",
        "MU",
        "m",
        "P",
        "tribes_num",
        "initial_N",
        "N_mean_tribe_size",
        "Nm",
        "initial haplotype_diversity",
        "haplotype_diversity",
        "haplotype_reduction",
        "initial nucleotide_diversity",
        "nucleotide_diversity",
        "nucleotide_reduction",
        "cult_divergence",
        "initial genetic_divergence",
        "initial hs",
        "initial ht",
        "genetic_divergence",
        "hs",
        "ht",
        "ev_freq",
        "ev_mag",
        "ev_effect",
        "only_pos",
        "assi_freq",
        "assi_mag",
        "assim_vs_innov",
    )

    def writer(header, filename):
        os.makedirs(filename.parent, exist_ok=True)
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

    writer(header, filename)


def update_file(fields):
    ## update result csv
    filename = RES_DIR / "full_results.csv"
    with open(filename, "a") as f:
        writer = csv.writer(f)
        writer.writerow(fields)


def save_results(soc: state.Society):
    # Add new society's information to results csv
    hap_n = soc.haplotype_count()
    res = [
        soc.K,
        soc.myu,
        soc.m,
        soc.p,
    ]

    # n_tribes
    n_tribes = len(soc.tribes)
    res.append(n_tribes)
    res.append(soc.initial_N)

    # mean_n
    mean_n = soc.total_size() / n_tribes
    res.append(mean_n)
    # migrates TODO now its only estimated num, need to change?
    res.append(mean_n * soc.m)

    # h_diversity
    res.append(soc.initial_haplotype_diversity)
    res.append(haplotype_diversity(soc, hap_n))
    res.append(reduction_percent(soc, "haplotype"))

    # nucleotide diversity
    res.append(soc.initial_nucleotide_diversity)
    res.append(nucleotide_diversity(soc, hap_n))
    res.append(reduction_percent(soc, "nucleotide"))

    # coefficient of variation of w(c):
    res.append(cv_fitness(soc))

    # genetic divergence among tribes (GST)
    res.append(soc.initial_GST)
    res.append(soc.initial_Hs)
    res.append(soc.initial_Ht)
    gst, hs, ht = Genetic_Divergence_GST(soc, hap_n)
    res.append(gst)
    res.append(hs)
    res.append(ht)

    # cultural innovation

    res.append(soc.ev_freq)
    res.append(soc.ev_mag)
    if soc.ev_freq != 0:
        res.append(float(soc.ev_mag * soc.ev_freq))
    else:
        res.append(0)

    res.append(soc.only_pos)

    res.append(soc.assi_freq)
    res.append(soc.assi_mag)
    # if there's no innovation effect their can be no assimilation effect

    if soc.innovation_effect > 0:
        assimilation_effect = float(soc.assimilation_effect / soc.innovation_effect)
    else:
        assimilation_effect = 0

    res.append(assimilation_effect)

    # open the file in  write mode
    filename = RES_DIR / "full_results.csv"
    with open(filename, "a") as f:
        writer = csv.writer(f)
        writer.writerow(res)

    update_file(res)

    return None


def extract_results():
    # Load  data from csv to pandas df

    dtype_dict = {
        "K": "int",
        "MU": "float64",
        "m": "float64",
        "P": "int",
        "tribes_num": "int",
        "initial_N": "float64",
        "N_mean_tribe_size": "float64",
        "Nm": "float64",
        "initial haplotype_diversity": "float64",
        "haplotype_diversity": "float64",
        "haplotype_reduction": "float64",
        "initial nucleotide_diversity": "float64",
        "nucleotide_diversity": "float64",
        "nucleotide_reduction": "float64",
        "cult_divergence": "float64",
        "initial genetic_divergence": "float64",
        "initial hs": "float64",
        "initial ht": "float64",
        "genetic_divergence": "float64",
        "final hs": "float64",
        "final ht": "float64",
        "ev_freq": "float64",
        "ev_mag": "float64",
        "ev_effect": "float64",
        "only_pos": "int",
        "assi_freq": "float64",
        "assi_mag": "float64",
        "assim_vs_innov": "float64",
        "assim_vs_innov": "int",
    }

    path_2 = RES_DIR

    df_results = pd.read_csv(path_2 / "full_results.csv", low_memory=False)

    return df_results


############## Fig 5 data from csv  ####################################################################################################################################################################################################################################


def fig_5_xaxis(id):
    ## each sun list is a row in figure 5, it will contain tuples of range

    if id == 0 or id == 1:
        # A -10 bars 0-1000
        limits = [
            0.001,
            0.01,
            0.042,
            0.178,
            0.750,
            3.162,
            13.335,
            56.234,
            237.137,
            1000,
        ]
        col_name = "initial_Nm"
    if id == 2 or id == 3:
        # B- 12 bars 0.000001- 0.02
        limits = [
            0.000001,
            0.0000023,
            0.0000053,
            0.0000123,
            0.0000285,
            0.0000658,
            0.0001520,
            0.0003511,
            0.0008111,
            0.0018738,
            0.0043288,
            0.0100000,
        ]
        col_name = "ev_effect"
    if id == 4 or id == 5:
        # c - 0.01- 10 ,10 bars
        limits = [
            0.01,
            0.021544346900318832,
            0.046415888336127774,
            0.1,
            0.21544346900318834,
            0.46415888336127775,
            1.0,
            2.154434690031882,
            4.6415888336127775,
            10.0,
        ]
        col_name = "assim_vs_innov"
    if id == 6 or id == 7:
        # D - 10 bars, 0-1
        limits = [
            0.001,
            0.0031622776601683794,
            0.01,
            0.03162277660168379,
            0.1,
            0.31622776601683794,
            1.0,
            3.1622776601683795,
            10.0,
        ]
        col_name = "cult_divergence"
    if id == 8 or id == 9:
        # E- 10 bars 0-1
        limits = [
            0.001,
            0.0031622776601683794,
            0.01,
            0.03162277660168379,
            0.1,
            0.31622776601683794,
            1.0,
            3.1622776601683795,
            10.0,
        ]
        col_name = "genetic_divergence"

    return col_name, limits


def adjust_vals(res, y_lim):
    n = len(res[0])
    for i in range(n):
        total_high = res[2][i] + res[1][i] + res[0][i]
        new_high = y_lim[1]
        if total_high > y_lim[1]:
            res[2][i] = (res[2][i] / total_high) * new_high
            res[1][i] = (res[1][i] / total_high) * new_high
            res[0][i] = (res[0][i] / total_high) * new_high

    return res


def fig_5_rows(full_data, id):
    col_name, x_limits = fig_5_xaxis(id)

    res = [[0 for _ in range(len(x_limits))] for _ in range(3)]
    if id % 2 == 0:
        reduc = "haplotype_reduction"
    else:
        reduc = "nucleotide_reduction"

    low_b = 0
    for i in range(len(x_limits)):
        up_b = x_limits[i]

        res[2][i] += full_data[
            (full_data[col_name] >= low_b)
            & (full_data[col_name] <= up_b)
            & (full_data[reduc] > 90)
        ].shape[0]
        res[1][i] += full_data[
            (full_data[col_name] >= low_b)
            & (full_data[col_name] <= up_b)
            & (full_data[reduc] < 90)
            & (full_data[reduc] > 50)
        ].shape[0]
        res[0][i] += full_data[
            (full_data[col_name] >= low_b)
            & (full_data[col_name] <= up_b)
            & (full_data[reduc] < 50)
        ].shape[0]

        low_b = up_b

    y_borders = [
        (0, 5000),
        (0, 5000),
        (0, 4000),
        (0, 4000),
        (0, 200),
        (0, 200),
        (0, 10000),
        (0, 10000),
        (0, 5000),
        (0, 5000),
    ]
    x_borders = [
        (0, 1000),
        (0, 1000),
        (0.000001, 0.02),
        (0.000001, 0.02),
        (0.01, 10),
        (0.01, 10),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
    ]
    res = adjust_vals(res, y_borders[id])
    return res, x_limits, y_borders[id], x_borders[id], col_name


if __name__ == "__main__":
    pass
