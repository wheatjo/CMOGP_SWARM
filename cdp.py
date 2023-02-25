from collections import defaultdict
from itertools import chain
from operator import attrgetter

def cdp_selNSGA2(individuals, k, nd='standard'):

    if nd == 'standard':
        pareto_fronts = sortNondominated(individuals, k, first_front_only = False)
    else:
        raise Exception('selNSGA2: The choice of non-dominated sorting '
                        'method "{0}" is invalid.'.format(nd))

    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        assignCrowdingDist(pareto_fronts[-1])
        sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
        chosen.extend(sorted_front[:k])

    return chosen

def sortNondominated(individuals, k, first_front_only = False):

    if k == 0:
        return []

    map_fit_ind = defaultdict(list) #存放fit相同的个体

    for ind in individuals:
        ind.fit = (ind.cv, ind.fitness.values)  #(cv值,(fitness,nodes))
        map_fit_ind[ind.fit].append(ind)
    fits = list(map_fit_ind.keys())   #元素是ind.fitness

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int) #被欺负的个数
    dominated_fits = defaultdict(list) #统治的fitness

    # Rank first Pareto front
    for i, fit_i in enumerate(fits):#fit_i是fitness
        for fit_j in fits[i+1:]:
            if dominate(fit_i, fit_j):
                dominating_fits[fit_j] += 1  #被支配的个数 +1
                dominated_fits[fit_i].append(fit_j)
            elif dominate(fit_j, fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:  #被支配的个数是0
            current_front.append(fit_i)

    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    #fronts[-1]是current_front的   ind集合形成的list
    pareto_sorted = len(fronts[-1])

    # Rank the next front until all individuals are sorted or
    # the given number of individual are sorted.
    if not first_front_only:
        N = min(len(individuals), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts

def assignCrowdingDist(individuals):

    if len(individuals) == 0:
        return

    distances = [0.0] * len(individuals)
    crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]

    nobj = len(individuals[0].fitness.values)

    for i in range(nobj):#对每个目标
        crowd.sort(key=lambda element: element[0][i])
        distances[crowd[0][1]] = float("inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (next[0][i] - prev[0][i]) / norm

    for i, dist in enumerate(distances):
        individuals[i].fitness.crowding_dist = dist

#严格支配
def dominate(ind_i,ind_j):#ind_i是否支配ind_j
    if ind_i[0] < ind_j[0]:  #cv值是第一个元素
        return True
    elif ind_i[0] == ind_j[0] and ind_i[1] < ind_j[1]:
        return True
    else:
        return False
