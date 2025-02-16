import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import time
import save_result
import copy
import state
import os
import pickle
from datetime import datetime
from pathlib import Path
import itertools
import shutil
import multiprocessing
import math
import pandas as pd


# create path
CODE_DIR = Path(__file__).parent
DATA_DIR = CODE_DIR.parent / "data"
SOCIETIES_DIR = DATA_DIR / "general" / "Base_line" / "socs"


## generation's steps #############################################################################################


def reproduction(soc: state.Society):
    # denominator of the fraction-Sum of the product of the size of each tribe times its fitness
    denominator = sum(c.w * c.size() for c in soc.tribes)

    empty_tribes = []  # will be deleted later
    for c in soc.tribes:
        del_h = []  # will be deleted later

        for h, n in c.n_h.items():
            # n is n(t,h,c)
            assert n > 0
            x_numerator = n * c.w * soc.K
            x = x_numerator / denominator

            # update n(t,h,c) for generation t+1.It is calculated from  Poason distribution  where x is the mean value.
            c.n_h[h] = np.random.poisson(x)

            if c.n_h[h] == 0:
                del_h.append(h)

        # delete tribes/haplotypes that got emptied
        for h in del_h:
            c.n_h.pop(h)
        if not c.n_h:
            empty_tribes.append(c)
    for tribe in empty_tribes:
        soc.tribes.remove(tribe)


def mutation(soc: state.Society):
    ## This func execute all mutation events in a generation given myu.
    ## The function uses a helper function soc.mutation_event()

    for c in soc.tribes:
        # new_hs is a dict initialized for every tribe separately,
        # containing for every haplotype key a value n- the number of mutation events from it.
        new_hs = {}

        for h, num_h in c.n_h.items():
            # n is the number of mutation events in this generation for tribe c with haplotype h
            n = np.random.binomial(num_h, soc.myu)

            if n != 0:
                new_hs[h] = n

        # Helper function will create all new mutations for tribe c
        soc.mutation_event(new_hs, c)

    # deleting haplotypes that were extinct from society's haplotypes dict
    pop_h = []
    for h in soc.haplotypes:
        if all(h not in t.n_h for t in soc.tribes):
            pop_h.append(h)

    for h in pop_h:
        soc.haplotypes.pop(h)


def cultural_evolution(
    soc: state.Society, frequency: float, magnitude: float, only_pos=0
):
    ## the function calculates the effect of cultural innovations on each tribe's fitness in the society soc,
    # only pos = 0: the effect can be both positive or negative or just positive if only pos =1.

    for c in soc.tribes:
        # check if cultural innovation will occur in tribe c in current generation :
        if random.random() <= frequency:
            # calculate the new fitness N(1,magnitude) represents a normal random variable with mean 1.0 and standard
            # deviation - magnitude.

            # update new fitness according to calculated fitness_effect
            if only_pos == 1:
                fitness_effect = np.random.normal(0.0, magnitude)
                new_w = c.w * (1 + abs(fitness_effect))
            else:
                # fitness_effect < 1 indicates a negative effect on tribe's fitness
                fitness_effect = np.random.normal(1.0, magnitude)
                new_w = abs(c.w * fitness_effect)

            soc.innovation_effect += abs(new_w - c.w)
            c.w = new_w


def cultural_assimilution(soc: state.Society, pr: float, magnitude: float):
    ##  the function receives:  society soc, probability pr of receiving
    ## cultural input from another random tribe and magnitude- the effect size of the assimilation on the receiving tribe.

    if len(soc.tribes) == 1:
        return None

    for c in soc.tribes:
        # If Assimilution occurs:
        if random.random() <= pr:
            # choose a tribe d to imitate:
            while True:
                d = random.choice(soc.tribes)
                if d is not c:
                    break

            new_w = c.w * (1 - magnitude) + d.w * magnitude
            soc.assimilation_effect += abs(new_w - c.w)
            c.w = new_w


def intertribe_migration(m: float, soc: state.Society):

    ## this func represents individuals migration between tribes. proportion of migration is m ,society soc
    ## each one will migrate to a randomly chosen tribe(could also be their original tribe) and immediately get the tribe's fitness.

    # tribes or haplotypes that were emptied. will be deleted later.
    empty_tribes = []

    for c in soc.tribes:
        # choose m individuals from tribe c
        c_size = c.size()

        n_leave = math.floor(m * c_size)
        if n_leave == 0:
            continue
        # Randomly choosing n_leave individuals to leave:
        leaving = random.sample(range(1, c_size + 1), n_leave)
        leaving.sort()

        # update c.n_h and convert leaving into haplotypes list
        # looping over the dictionary ,summing values ("each individual has a number")

        # create sum list:
        sum = 0
        sum_list = []
        for h in c.n_h:
            if c.n_h[h] > 0:
                sum += c.n_h[h]
                sum_list.append((h, sum))

        # haplotypes that were emptied. will be deleted later.
        empty_hs = []
        h_idx = 0

        # each iteration is a single migration event
        for id in range(n_leave):
            while leaving[id] > sum_list[h_idx][1]:
                h_idx += 1

            # converting individual's number to his haplotype
            h = sum_list[h_idx][0]
            leaving[id] = h

            # update tribe's data
            c.n_h[h] -= 1

            assert c.n_h[h] >= 0
            if c.n_h[h] == 0:
                empty_hs.append(h)

        for empty_h in empty_hs:
            c.n_h.pop(empty_h)
        if c.size() == 0:
            empty_tribes.append(c)
        assert c.size() > 0

        # now leaving is a list of haplotypes which will be randomly assigned to new tribes:
        for h in leaving:
            d = random.choice(soc.tribes)
            d.n_h[h] = d.n_h.get(h, 0) + 1

    for empty_tribe in empty_tribes:
        soc.tribes.remove(empty_tribe)


def tribe_fission(soc: state.Society, P: float):
    ## Fission of a tribe is when its split into two separate new tribes. the func receives society soc and P-splitting parameter
    ## and calculates prob (in paper q(c,t))- chance of splitting depends on the tribe's size.
    ## if fission occur proportion (p) of individuals with each haplotype will move to a new tribe. TODO

    general_density = soc.K * P
    # for each tribe c in the society
    new_tribes = []
    empty_tribes = []  # tribes that will get empty

    for c in soc.tribes:

        # calculating prob
        prob = c.size() / (general_density + c.size())

        # If splitting occurs
        if random.random() <= prob:

            # choosing randomly the proportion to leave (p in article) from the [O 1] uniform distribution.
            proportion = np.random.uniform(0, 1)
            # creating the new tribe
            new_tribe = state.Tribe(c.w)

            empty_hs = []
            for h in c.n_h:
                n_leave = int(c.n_h[h] * proportion)
                if n_leave == 0:
                    continue
                c.n_h[h] -= n_leave
                if c.n_h[h] <= 0:
                    empty_hs.append(h)
                new_tribe.n_h[h] = n_leave
            # update tribe's data
            for h_drop in empty_hs:
                c.n_h.pop(h_drop)

            if new_tribe.size() > 0:
                new_tribes.append(new_tribe)

            if c.size() == 0:
                empty_tribes.append(c)

    for empty_tribe in empty_tribes:
        soc.tribes.remove(empty_tribe)
    soc.tribes.extend(new_tribes)


############ full simulation runs ##############################################################################################################################################################################################
## measuring run time


def generation_simulation(
    soc: state.Society,
    ev_freq=None,
    ev_magnitude=None,
    only_pos=None,
    assimilation_pr=None,
    Assimilation_mag=None,
):
    ## running all events by order : reproduction, genetic mutation, cultural evolution,
    ## cultural assimilation, intertribe migration, and tribe fission

    # reproduction phase:
    reproduction(soc)

    # Mutation  phase:
    mutation(soc)

    # cultural evolution:
    if ev_freq is not None:
        cultural_evolution(soc, ev_freq, ev_magnitude, only_pos)

        # cultural assimilation (only relevant if cultural evolution occur):
        if assimilation_pr is not None:
            cultural_assimilution(soc, assimilation_pr, Assimilation_mag)

    # intertribe migration:
    if soc.m > 0:
        intertribe_migration(soc.m, soc)

    # tribe fission:
    tribe_fission(soc, soc.p)

    # Moving to the next generation
    soc.generation += 1

    return None


def baseline_population(
    environment_capacity: int, mutation_myu: float, fission_p: float, migration_m: float
) -> state.Society:
    ## the function will create 1 society combined of K identical individuals and will run 2 * TE generations without any cultural effects,
    ## where TE is the number og generations needed until  genetic diversity first exceeded that expected from the infinite allele model

    # calculating  expected diversity from the infinite allele model
    expected_numerator = 2 * mutation_myu * environment_capacity
    infinite_allele_expected = expected_numerator / (1 + expected_numerator)

    # creating the new society, K is also the society's initial size.
    soc = state.Society(
        environment_capacity,
        mutation_myu,
        fission_p,
        migration_m,
        environment_capacity,
    )

    # finding TE
    while True:
        generation_simulation(soc)
        diversity = save_result.haplotype_diversity(soc, soc.haplotype_count())
        if diversity > infinite_allele_expected:
            break

    # run another TE generations
    te = soc.generation
    for _ in range(te):
        generation_simulation(soc)

    # save initial cultural structure
    hap_count = soc.haplotype_count()
    soc.initial_haplotype_diversity = save_result.haplotype_diversity(soc, hap_count)
    soc.initial_nucleotide_diversity = save_result.nucleotide_diversity(soc, hap_count)
    soc.initial_GST, soc.initial_Hs, soc.initial_Ht = (
        save_result.Genetic_Divergence_GST(soc, hap_count)
    )
    soc.initial_N = soc.total_size() / len(soc.tribes)
    return soc


def table_one_helper(args):
    k, myu, m, p = args
    socs = [baseline_population(k, myu, p, m) for _ in range(10)]
    return socs


def table_one_combos():
    # Creates a list of baseline societies: 10 * each table 1 combinations.
    all_params = []
    for k in [1000]:
        for myu in [0.001]:
            for m in [0.1, 0.01, 0.001, 0.0001, 0]:
                for p in [1, 10, 100, 1000]:
                    all_params.append((k, myu, m, p))
    for k in [10000]:
        for myu in [0.001, 0.0001]:
            for m in [0.1, 0.01, 0.001, 0.0001, 0]:
                for p in [1, 10, 100, 1000]:
                    all_params.append((k, myu, m, p))
    for k in [100000]:
        for myu in [0.0001, 0.00001]:
            for m in [0.01, 0.001, 0.0001, 0]:
                for p in [100, 1000]:
                    all_params.append((k, myu, m, p))

    base_soc_list = []
    for params in all_params:
        base_soc_list.extend(table_one_helper(params))

    return base_soc_list


def p2_simulation(
    soc_list, ev_freq=None, ev_mag=None, only_pos=None, ass_pr=None, ass_mag=None
):

    for soc in soc_list:
        # update soc values
        soc.ev_freq = ev_freq
        soc.ev_mag = ev_mag
        soc.only_pos = only_pos
        soc.assi_freq = ass_pr
        soc.assi_mag = ass_mag

        for _ in range(200):
            generation_simulation(soc, ev_freq, ev_mag, only_pos, ass_pr, ass_mag)

    return None


def assimilation_base_line(base_soc_list):
    # receive table 1 data,returns only relevant
    soc_ass_list = []
    for soc in base_soc_list:
        if soc.K == 1000:
            if soc.myu == 0.001:
                if soc.p == 1:
                    if soc.m == 0.0001:
                        soc_ass_list.append(copy.deepcopy(soc))
        elif soc.K == 10000:
            if soc.myu == 0.001:
                if soc.p == 1:
                    if soc.m == 0.0001:
                        soc_ass_list.append(copy.deepcopy(soc))

            elif soc.myu == 0.0001:
                if soc.p == 1:
                    if soc.m == 0.0001:
                        soc_ass_list.append(copy.deepcopy(soc))
        else:
            if soc.m == 0.0001:
                if soc.p == 100:
                    if soc.myu == 0.00001 or soc.myu == 0.0001:
                        soc_ass_list.append(copy.deepcopy(soc))

    return soc_ass_list


def cultural_effect_helper(base_soc_list, soc_ass_list, params):
    ev_frq = params[0]
    ev_mag = params[1]
    pos = params[2]

    # runs with no assimilation
    if len(params) == 3:
        soc_copy_list = [copy.deepcopy(soc) for soc in base_soc_list]
        p2_simulation(soc_copy_list, ev_frq, ev_mag, pos)

        # save to cls
        for soc in soc_copy_list:
            save_result.save_results(soc)

    else:
        ass_frq = params[3]
        ass_mag = params[4]
        ass_copy_list = [copy.deepcopy(soc) for soc in soc_ass_list]
        p2_simulation(ass_copy_list, ev_frq, ev_mag, pos, ass_frq, ass_mag)

        # save to cls
        for soc in ass_copy_list:
            save_result.save_results(soc)

    return None


def cultural_effect():
    base_soc_list = table_one_combos()
    soc_list = [copy.deepcopy(soc) for soc in base_soc_list]

    # control group
    p2_simulation(soc_list)
    (save_result.save_results(soc) for soc in soc_list)

    cult_params = []
    # with cultural_evolution but no cultural_assimilution:
    for b1 in [1, 0]:
        for freq in [0.1, 0.01, 0.001, 0.0001]:
            for mag in [0.0125, 0.05, 0.2]:
                cult_params.append((freq, mag, b1))
    # with assimilation
    for ass_mag in [0.2, 0.5, 0.8]:
        for ass_pr in [0, 0.01, 0.05, 0.1, 0.5]:
            for b1 in [1, 0]:
                for freq in [0.1, 0.01, 0.001, 0.0001]:
                    for mag in [0.0125, 0.05, 0.2]:
                        cult_params.append((freq, mag, b1, ass_pr, ass_mag))

    # create new soc list only relevant for assimilation runs
    soc_ass_list = assimilation_base_line(base_soc_list)

    for param in cult_params:
        cultural_effect_helper(base_soc_list, soc_ass_list, param)
    return None


## Non-heritable Demographic Variation ###############################################################
def fit_change_table_2(soc, bool_Siemann, variable):
    if bool_Siemann:
        for c in soc.tribes:
            if random.random() < variable:
                soc.tribes.remove(c)
            else:
                c.w = 1

    else:
        for c in soc.tribes:
            c.w = np.random.normal(1.0, variable)
            if 0 >= c.w:

                soc.tribes.remove(c)

    return None


def data_table_2():
    # plot demographic effects
    data_for_plot = [[0, 0, 0, 0] for _ in range(9)]
    base_soc_list = table_one_combos()

    # first 4 are v vals
    fit_changes = [0.05, 0.1, 0.2, 0.4, 0.0125, 0.025, 0.05, 0.1]
    soc_list = [copy.deepcopy(soc) for soc in base_soc_list]

    ## control group
    p2_simulation(soc_list)

    for soc in soc_list:
        if soc.total_size() == 0:
            data_for_plot[0][3] += 1
        else:
            reduction_i = save_result.reduction_percent_951(soc, "haplotype")
            data_for_plot[0][reduction_i] += 1

    # next rows:
    for i in range(len(fit_changes)):
        soc_list = [copy.deepcopy(soc) for soc in base_soc_list]
        for soc in soc_list:

            for j in range(200):
                generation_simulation(soc)
                if i > 3:
                    # under Siemann_model
                    fit_change_table_2(soc, True, fit_changes[i])
                else:
                    # under Tiedemann_Milinkovitch_model model
                    fit_change_table_2(soc, False, fit_changes[i])

                if len(soc.tribes) == 0:
                    data_for_plot[i + 1][3] += 1
                    break

            if len(soc.tribes) != 0:
                reduction_i = save_result.reduction_percent_951(soc, "haplotype")
                data_for_plot[i + 1][reduction_i] += 1

    return data_for_plot


###### plots data #################################################################################################


def baseline_fig_one_or_two():
    environment_capacity = 10000
    mutation_myu = 0.0001
    fission_p = 1
    migration_m = 0.0001
    soc_list = []

    for _ in range(10):
        soc_list.append(
            baseline_population(
                environment_capacity, mutation_myu, fission_p, migration_m
            )
        )

    return soc_list


def cult_figure_1_or_2(soc: state.Society, ev_freq, only_pos=1):
    ev_mag = 0.05

    # update soc values
    soc.ev_freq = ev_freq
    soc.ev_mag = ev_mag
    soc.only_pos = only_pos

    hap_diversity = []  # len=200 diversity measures for 1 soc
    cultural_divergence = []  # len=200 CV measures for 1 soc

    for _ in range(200):
        generation_simulation(soc, ev_freq, ev_mag, only_pos)

        # save_results:
        hap_diversity.append(
            save_result.haplotype_diversity(soc, soc.haplotype_count())
        )

        # coefficient of variation of w(c):
        cultural_divergence.append(save_result.cv_fitness(soc))

    return cultural_divergence, hap_diversity


def figure_1_Data():
    ev_freq = 0.001

    # each list will contain 10 lists in len 200
    cultural_divergences = []
    hap_diversities = []

    soc_list = baseline_fig_one_or_two()

    for soc in soc_list:
        soc_cultural_divergence, soc_hap_diversity = cult_figure_1_or_2(soc, ev_freq)
        cultural_divergences.append(soc_cultural_divergence)
        hap_diversities.append(soc_hap_diversity)

    res = (hap_diversities, cultural_divergences)
    for soc in soc_list:
        save_result.save_results(soc)

    return res


def figure_2_data():
    soc_list = baseline_fig_one_or_two()

    # each empty list will contain 10 lists in len 200 each
    frq_divers = []
    partial_list = []
    ev_frqs = [0.0001, 0.001, 0.1]

    for ev_freq in ev_frqs:
        mutant_soc_list = [copy.deepcopy(soc) for soc in soc_list]
        for soc in mutant_soc_list:
            # hap_diversity is a 200 len list
            _, hap_diversity = cult_figure_1_or_2(soc, ev_freq)
            partial_list.append(copy.deepcopy(hap_diversity))
        # partial_list len is 10 of 200 len lists
        frq_divers.append(copy.deepcopy(partial_list))
        partial_list = []

    for soc in soc_list:
        save_result.save_results(soc)
    return frq_divers


def fig_2_mutations():
    ##"I checked to see whether cultural hitchhiking could result from
    ## purely negative innovations by making runs with the parameter
    ## combinations used in Figure 2,and just negative innovations."
    # this function is used to check diversity reduction in case of only negative vs. only positive innovations
    soc_list = baseline_fig_one_or_two()
    n = len(soc_list)
    frequencies = [0.0001, 0.001, 0.1]
    reductions = [
        [0, 0, 0],
        [0, 0, 0],
    ]  # reductions[0] shows the sum reductions of each of the three frequencies

    for fr in range(len(frequencies)):
        copy_lst_neg = [copy.deepcopy(soc) for soc in soc_list]
        copy_lst_pos = [copy.deepcopy(soc) for soc in soc_list]

        # only negative for a certain innovation frequency
        for soc in copy_lst_neg:
            # hap_diversity is a 200 len list
            reductions[0][fr] += cult_only_neg(soc, frequencies[fr])

        reductions[0][fr] /= n

        # only positive for a certain innovation frequency
        p2_simulation(copy_lst_pos, frequencies[fr], 0.05, 1)

        for soc in copy_lst_pos:

            reductions[1][fr] += save_result.reduction_percent(soc, "haplotype")

        reductions[1][fr] /= n

        # Create pandas df to present a table

    column_titles = ["frequency = 0.0001", "frequency = 0.001", "frequency = 0.1"]
    row_titles = ["Purely negative model", "Purely positive model"]

    # Create a DataFrame
    df = pd.DataFrame(reductions, columns=column_titles, index=row_titles)
    print(df)


def neg_evolution(soc: state.Society, frequency: float, magnitude: float, only_pos=0):

    for c in soc.tribes:
        # check if cultural innovation will occur in tribe c in current generation :
        if random.random() <= frequency:
            # calculate the new fitness N(1,magnitude) represents a normal random variable with mean 1.0 and standard
            # deviation - magnitude.
            # update new fitness according to calculated fitness_effect
            fitness_effect = np.random.normal(0.5, magnitude)
            fitness_effect = np.clip(fitness_effect, 0, 1)
            new_w = c.w * fitness_effect
            soc.innovation_effect += abs(new_w - c.w)
            c.w = new_w


def only_neg_generation_simulation(
    soc: state.Society,
    ev_freq=None,
    ev_magnitude=None,
):
    reproduction(soc)
    mutation(soc)

    # cultural evolution:
    neg_evolution(soc, ev_freq, ev_magnitude)

    # intertribe migration:
    if soc.m > 0:
        intertribe_migration(soc.m, soc)
    # tribe fission:
    tribe_fission(soc, soc.p)
    # Moving to the next generation
    soc.generation += 1

    return None


def cult_only_neg(soc: state.Society, ev_freq):
    ev_mag = 0.05
    # update soc values
    soc.ev_freq = ev_freq
    soc.ev_mag = ev_mag
    soc.only_pos = -1

    for _ in range(200):
        only_neg_generation_simulation(soc, ev_freq, ev_mag)

    return save_result.reduction_percent(soc, "haplotype")


def fig_3_data():
    base_soc = fig_3_baseline()

    data = []
    for co in range(len(base_soc)):
        ps = len(base_soc[co])
        ps_sum = [[[0 for p in range(ps)] for _ in range(3)] for _ in range(2)]
        data.append(copy.deepcopy(ps_sum))

    # start running model
    for i in range(2000):
        for co in range(5):
            for p in range(len(base_soc[co])):
                soc = copy.deepcopy(base_soc[co][p])
                hap_reduction, nuc_reduction = model_run_fig_3(soc)
                data[co][0][hap_reduction][p] += 1
                data[co][1][nuc_reduction][p] += 1

    return data


def fig_3_baseline():
    # create baseline societies
    p1 = [1, 10, 100, 1000]
    p2 = [1000, 10000]

    combo_1 = []
    combo_2 = []
    combo_3 = []
    combo_4 = []
    combo_5 = []

    # create base line populations:

    for p in p1:
        combo_1.append(baseline_population(1000, 0.001, p, 0.01))
        combo_2.append(baseline_population(10000, 0.001, p, 0.01))
        combo_3.append(baseline_population(10000, 0.0001, p, 0.01))

    for p in p2:
        combo_4.append(baseline_population(100000, 0.0001, p, 0.01))
        combo_5.append(baseline_population(100000, 0.00001, p, 0.01))

    soc_list = [combo_1, combo_2, combo_3, combo_4, combo_5]

    return soc_list


def model_run_fig_3(soc):

    p2_simulation([soc], 0.1, 0.2)

    save_result.save_results(soc)
    hap_reduction = save_result.reduction_percent_951(soc, "haplotype")
    nuc_reduction = save_result.reduction_percent_951(soc, "nucleotide")

    return (hap_reduction, nuc_reduction)


if __name__ == "__main__":
    cultural_effect()
