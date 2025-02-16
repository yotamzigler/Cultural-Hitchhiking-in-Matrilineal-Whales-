import copy


class Society:
    ## Society contains all existing tribes in an environment with capacity k.

    def __init__(self, capacity, myu, p, m, size):
        self.generation = 0
        self.K = capacity  # capacity of environment
        self.p = p
        self.m = m
        self.myu = myu
        # haplotypes is a dictionary for each haplotype existing in the society it holds in it's value a list representing of the "genotype" of said haplotype.
        self.haplotypes = {
            0: [0]
        }  # When starting a new society there will be one type of haplotype,
        self.ev_freq = 0
        self.ev_mag = 0
        self.only_pos = 0  # 0 : innovations for this society can have either positive or negative effect, 1: only positive , -1 : only negative
        self.assi_freq = 0
        self.assi_mag = 0
        self.initial_haplotype_diversity = None
        self.initial_nucleotide_diversity = None
        self.initial_GST = None
        self.initial_Hs = None
        self.initial_Ht = None
        self.initial_N = 0  # Mean tribe size

        # Society's tribes
        first_tribe = Tribe()
        first_tribe.n_h = {
            0: size
        }  # Number of individuals with each haplotype in the tribe

        self.tribes = [first_tribe]  # starting with one big tribe

        # Sum of fitness changes (in absolute value) in society:
        self.assimilation_effect = 0
        self.innovation_effect = 0

    def total_size(self):
        # Return the total society size
        return sum(sum(c.n_h.values()) for c in self.tribes)

    def haplotype_count(self):
        # Return a dictionary of the number of individuals with each haplotype in the society
        h_vals = {}
        for c in self.tribes:
            for h in c.n_h:
                h_vals[h] = h_vals.get(h, 0) + c.n_h[h]

        return h_vals

    def mutation_event(self, new_hs, tribe_c):
        # This function is called for every tribe's mutation phase. It receive's a tribe- tribe_c and,
        # "new_hs", a dictionary with keys that present from which original haplotype the mutation occurs and value,
        #  how many individuals with said original haplotype had been mutated.

        new_haplotype = max(list(self.haplotypes.keys())) + 1

        for h, n_mute in new_hs.items():
            for _ in range(n_mute):
                original_seq = self.haplotypes[h]
                new_seq = copy.deepcopy(original_seq)
                new_seq.append(new_haplotype)
                self.haplotypes[new_haplotype] = new_seq

                tribe_c.n_h[new_haplotype] = 1

                new_haplotype += 1

            # update tribe's h-list
            tribe_c.n_h[h] -= n_mute
            if tribe_c.n_h[h] <= 0:
                tribe_c.n_h.pop(h)

        return None

    def calculate_bp_dif(self, h1, h2):
        # This function is used for nucleotide diversity calculations. it returns the number of different nucleotides between the two genotypes.
        len_h1 = len(self.haplotypes[h1])
        len_h2 = len(self.haplotypes[h2])

        if len_h1 >= len_h2:
            long_seq = self.haplotypes[h1]
            short_seq = self.haplotypes[h2]
        else:
            long_seq = self.haplotypes[h2]
            short_seq = self.haplotypes[h1]

        diff = len(long_seq) - len(short_seq)

        for i in range(len(short_seq)):
            if long_seq[i] != short_seq[i]:
                diff = len(long_seq) - i
                break

        return diff

    def __str__(self):
        # Print society's information.
        s = ""
        s += f"total soc size={self.total_size()}\nnum of tribes: {len(self.tribes)}\n{self.K=}{self.m=}{self.myu=}{self.p=}\n"
        for c in self.tribes:
            s += str(c)
            s += "\n"
        return s


class Tribe:
    def __init__(self, fitness=1):

        self.w = fitness  # w(c,t), only effected by cultural evolution
        self.n_h = dict()  # Number of individuals with each haplotype in the tribe

    def size(self):
        # Return the tribe's size.
        return sum(self.n_h.values())

    def __str__(self):
        # Print tribe's information.
        return f"{self.w=}{self.size()=}{self.n_h=}"
