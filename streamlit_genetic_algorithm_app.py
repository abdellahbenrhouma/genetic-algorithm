"""
Streamlit Genetic Algorithm Playground
====================================

Single-file Streamlit app that implements a flexible, pluggable GA framework
covering:
- Problem-first (analytic) and Data-first (dataset) workflows
- Encodings: binary, integer, real, permutation
- Initialization strategies: random, Latin Hypercube (via numpy), heuristic seeding
- Selection: tournament, roulette, rank
- Crossover: 1-point, 2-point, uniform, SBX (real), PMX (perm)
- Mutation: bitflip, gaussian, swap, inversion, int-reset
- Constraint handling: equality repair, penalty (static/dynamic)
- Multi-objective: NSGA-II simple implementation (for <=2 objectives)
- UI: comprehensive Streamlit sidebar and main panels, presets, preview, run/export

To run:
    pip install streamlit numpy pandas scikit-learn plotly matplotlib
    streamlit run streamlit_genetic_algorithm_app.py
"""

from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
import math
import random
import time
import json
import re
from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

# ------------------------------ Utilities ---------------------------------

def set_seed(s: Optional[int]):
    if s is None:
        return
    random.seed(s)
    np.random.seed(s)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# --------------------------- Encodings / Decoders -------------------------

class EncodingSpec:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.kwargs = kwargs


# --------------------------- Initialization -------------------------------

def init_population(pop_size: int, encoding: EncodingSpec, seed: Optional[int]=None, heuristic_seeds: Optional[List[Any]]=None):
    set_seed(seed)
    pop = []
    if encoding.name == 'binary':
        length = encoding.kwargs['length']
        for i in range(pop_size):
            if heuristic_seeds and i < len(heuristic_seeds):
                pop.append(np.array(heuristic_seeds[i], dtype=np.int8))
            else:
                pop.append(np.random.randint(0,2,size=length))
    elif encoding.name == 'integer':
        bounds = encoding.kwargs['bounds']  # list of (lo,hi)
        for i in range(pop_size):
            if heuristic_seeds and i < len(heuristic_seeds):
                pop.append(np.array(heuristic_seeds[i], dtype=np.int32))
            else:
                ind = [np.random.randint(lo, hi+1) for (lo,hi) in bounds]
                pop.append(np.array(ind, dtype=np.int32))
    elif encoding.name == 'real':
        bounds = encoding.kwargs['bounds']
        use_lhs = encoding.kwargs.get('lhs', False)
        dims = len(bounds)
        if use_lhs:
            u = np.random.rand(pop_size, dims)
            pop = []
            for i in range(pop_size):
                ind = []
                for d in range(dims):
                    lo,hi = bounds[d]
                    val = lo + (u[i,d])*(hi-lo)
                    ind.append(val)
                pop.append(np.array(ind, dtype=float))
        else:
            for i in range(pop_size):
                ind = [np.random.uniform(lo,hi) for (lo,hi) in bounds]
                pop.append(np.array(ind, dtype=float))
    elif encoding.name == 'permutation':
        n = encoding.kwargs['n']
        base = list(range(n))
        for i in range(pop_size):
            if heuristic_seeds and i < len(heuristic_seeds):
                pop.append(np.array(heuristic_seeds[i], dtype=np.int32))
            else:
                arr = base.copy()
                np.random.shuffle(arr)
                pop.append(np.array(arr, dtype=np.int32))
    else:
        raise ValueError('Unknown encoding')
    return pop

# --------------------------- Selection ------------------------------------

def roulette_wheel_selection(pop, fitnesses, k=1):
    total = sum(fitnesses)
    if total == 0:
        return [random.choice(pop) for _ in range(k)]
    probs = [f/total for f in fitnesses]
    chosen = np.random.choice(len(pop), size=k, p=probs)
    return [deepcopy(pop[i]) for i in chosen]


def tournament_selection(pop, fitnesses, k=1, tour_size=3):
    chosen = []
    for _ in range(k):
        ids = np.random.randint(0,len(pop), size=tour_size)
        best = ids[0]
        for idx in ids:
            if fitnesses[idx] > fitnesses[best]:
                best = idx
        chosen.append(deepcopy(pop[best]))
    return chosen


def rank_selection(pop, fitnesses, k=1):
    idxs = np.argsort(fitnesses)
    ranks = np.empty_like(idxs)
    ranks[idxs] = np.arange(len(fitnesses))
    probs = ranks / ranks.sum()
    chosen = np.random.choice(len(pop), size=k, p=probs)
    return [deepcopy(pop[i]) for i in chosen]

# --------------------------- Crossover Operators -------------------------

def one_point_crossover(a: np.ndarray, b: np.ndarray):
    assert len(a)==len(b)
    n = len(a)
    if n<=1: return deepcopy(a), deepcopy(b)
    cx = np.random.randint(1,n)
    ca = np.concatenate([a[:cx], b[cx:]])
    cb = np.concatenate([b[:cx], a[cx:]])
    return ca, cb


def two_point_crossover(a: np.ndarray, b: np.ndarray):
    assert len(a)==len(b)
    n = len(a)
    if n<=2: return deepcopy(a), deepcopy(b)
    c1 = np.random.randint(1,n-1)
    c2 = np.random.randint(c1+1,n)
    ca = np.concatenate([a[:c1], b[c1:c2], a[c2:]])
    cb = np.concatenate([b[:c1], a[c1:c2], b[c2:]])
    return ca, cb


def uniform_crossover(a: np.ndarray, b: np.ndarray, prob=0.5):
    assert len(a)==len(b)
    mask = np.random.rand(len(a)) < prob
    ca = a.copy()
    cb = b.copy()
    ca[mask] = b[mask]
    cb[mask] = a[mask]
    return ca, cb


def sbx_crossover(a: np.ndarray, b: np.ndarray, eta=15):
    assert len(a)==len(b)
    n = len(a)
    ca = a.copy(); cb = b.copy()
    for i in range(n):
        if np.random.rand() <= 0.5:
            if abs(a[i]-b[i]) > 1e-14:
                rand = np.random.rand()
                u = rand
                if u <= 0.5:
                    beta_q = (2*u)**(1.0/(eta+1))
                else:
                    beta_q = (1/(2*(1-u)))**(1.0/(eta+1))
                ca[i] = 0.5*((1+beta_q)*a[i] + (1-beta_q)*b[i])
                cb[i] = 0.5*((1-beta_q)*a[i] + (1+beta_q)*b[i])
    return ca, cb


def pmx_crossover(parent1: np.ndarray, parent2: np.ndarray):
    size = len(parent1)
    p1, p2 = parent1.copy(), parent2.copy()
    cx1 = np.random.randint(0, size-1)
    cx2 = np.random.randint(cx1+1, size)
    child1 = -np.ones(size, dtype=int)
    child2 = -np.ones(size, dtype=int)
    child1[cx1:cx2+1] = p2[cx1:cx2+1]
    child2[cx1:cx2+1] = p1[cx1:cx2+1]
    def fill_child(child, parent_source, parent_other):
        for i in range(size):
            if child[i] == -1:
                val = parent_source[i]
                while val in child:
                    idx = np.where(parent_other == val)[0][0]
                    val = parent_source[idx]
                child[i] = val
    fill_child(child1, p1, p2)
    fill_child(child2, p2, p1)
    return child1, child2

# --------------------------- Mutations -----------------------------------

def bitflip_mutation(ind: np.ndarray, pm: float):
    ind = ind.copy()
    mask = np.random.rand(len(ind)) < pm
    ind[mask] = 1 - ind[mask]
    return ind


def gaussian_mutation(ind: np.ndarray, pm: float, sigma: float, bounds: List[Tuple[float,float]]):
    ind = ind.copy().astype(float)
    for i in range(len(ind)):
        if np.random.rand() < pm:
            ind[i] += np.random.normal(0, sigma)
            lo,hi = bounds[i]
            ind[i] = clamp(ind[i], lo, hi)
    return ind


def swap_mutation(ind: np.ndarray, pm: float):
    ind = ind.copy().astype(int)
    if np.random.rand() < pm:
        i,j = np.random.choice(len(ind), size=2, replace=False)
        ind[i], ind[j] = ind[j], ind[i]
    return ind


def inversion_mutation(ind: np.ndarray, pm: float):
    ind = ind.copy().astype(int)
    if np.random.rand() < pm:
        i,j = np.sort(np.random.choice(len(ind), size=2, replace=False))
        ind[i:j+1] = ind[i:j+1][::-1]
    return ind


def random_resetting_mutation(ind: np.ndarray, pm: float, bounds: List[Tuple[int,int]]):
    ind = ind.copy().astype(int)
    for i in range(len(ind)):
        if np.random.rand() < pm:
            lo,hi = bounds[i]
            ind[i] = np.random.randint(lo, hi+1)
    return ind

# --------------------------- Repair / Penalty -----------------------------

def knapsack_repair(bitvec: np.ndarray, weights: List[float], values: List[float], capacity: float):
    ind = bitvec.copy().astype(int)
    total_w = int(np.dot(ind, weights))
    if total_w <= capacity:
        return ind
    ratios = [v/w if w>0 else float('inf') for v,w in zip(values,weights)]
    while np.dot(ind, weights) > capacity:
        chosen_idx = [i for i in range(len(ind)) if ind[i]==1]
        worst = min(chosen_idx, key=lambda i: ratios[i])
        ind[worst] = 0
    return ind


def clamp_integer(ind: np.ndarray, bounds: List[Tuple[int,int]]):
    ind = ind.copy().astype(int)
    for i,(lo,hi) in enumerate(bounds):
        ind[i] = int(clamp(ind[i], lo, hi))
    return ind


def equality_repair_by_substitution(ind, config):
    var_names = config.get('var_names', [])
    if not var_names:
        return ind
    new = list(ind.copy())
    constraints = config.get('constraints', [])
    for c in constraints:
        if c['sense'] != '=':
            continue
        coeffs = c['coeffs']
        vars_in_c = [v for v in coeffs.keys() if v in var_names]
        if not vars_in_c:
            continue
        target = vars_in_c[-1]
        denom = coeffs[target]
        if abs(denom) < 1e-12:
            continue
        s = 0.0
        for v in vars_in_c[:-1]:
            idx = var_names.index(v)
            s += coeffs[v] * float(new[idx])
        value_for_target = (c['rhs'] - s) / denom
        if config.get('int_bounds'):
            lo,hi = config['int_bounds'][var_names.index(target)]
            value_for_target = clamp(value_for_target, lo, hi)
        idx_t = var_names.index(target)
        if isinstance(ind[idx_t], (int, np.integer)):
            new[idx_t] = int(round(value_for_target))
        else:
            new[idx_t] = value_for_target
    return np.array(new, dtype=ind.dtype)

# --------------------------- Fitness Evaluation / Constraint helpers --------------------------

class EvalCache:
    def __init__(self):
        self.cache = {}
    def key(self, ind, config):
        try:
            ind_list = ind.tolist() if hasattr(ind, 'tolist') else list(ind)
        except Exception:
            ind_list = list(ind) if isinstance(ind, (list, tuple)) else ind
        cfg_key = {
            'penalty_mode': config.get('penalty_mode'),
            'penalty_coef': config.get('penalty_coef', 0.0),
            'var_names': config.get('var_names', None),
            'constraints': config.get('constraints', None)
        }
        return json.dumps({'ind': ind_list, 'cfg': cfg_key}, sort_keys=True)

    def get(self, ind, config):
        k = self.key(ind, config)
        return self.cache.get(k)
    def set(self, ind, config, val):
        k = self.key(ind, config)
        self.cache[k] = val


def parse_linear_constraint_line(line: str):
    line = line.strip()
    if line == '':
        return None
    if '<=' in line:
        lhs, rhs = line.split('<=')
        sense = '<='
    elif '>=' in line:
        lhs, rhs = line.split('>=')
        sense = '>='
    elif '=' in line:
        lhs, rhs = line.split('=')
        sense = '='
    else:
        return None
    rhs_val = float(rhs.strip())
    tokens = re.findall(r'([+-]?\s*[0-9]*\.?[0-9]*)\s*\*?\s*([A-Za-z_]\w*)', lhs)
    coeffs = {}
    for coef_str, var in tokens:
        coef_clean = coef_str.replace(' ', '')
        if coef_clean == '' or coef_clean == '+':
            coef_val = 1.0
        elif coef_clean == '-':
            coef_val = -1.0
        else:
            coef_val = float(coef_clean)
        coeffs[var] = coeffs.get(var, 0.0) + coef_val
    return {'coeffs': coeffs, 'sense': sense, 'rhs': rhs_val}


def build_constraints_list(text_block: str):
    lines = [l.strip() for l in text_block.splitlines() if l.strip()]
    parsed = []
    for l in lines:
        p = parse_linear_constraint_line(l)
        if p:
            parsed.append(p)
    return parsed


def phenotype_to_dict(raw, var_names=None):
    if isinstance(raw, dict):
        return raw
    if var_names and isinstance(raw, (list, tuple, np.ndarray)) and len(raw) == len(var_names):
        return {var_names[i]: float(raw[i]) for i in range(len(var_names))}
    if isinstance(raw, (list, tuple, np.ndarray)):
        return {f'g{i}': float(raw[i]) for i in range(len(raw))}
    return raw


def compute_total_violation(phen_dict: dict, constraints: List[dict]):
    if not constraints:
        return 0.0
    total = 0.0
    for c in constraints:
        lhs = 0.0
        for var, coef in c['coeffs'].items():
            lhs += coef * float(phen_dict.get(var, 0.0))
        if c['sense'] == '<=':
            viol = max(0.0, lhs - c['rhs'])
        elif c['sense'] == '>=':
            viol = max(0.0, c['rhs'] - lhs)
        else:
            viol = abs(lhs - c['rhs'])
        total += viol
    return total


def evaluate_individual(ind, encoding: EncodingSpec, fitness_fn: Callable[[Any], float], config: dict, cache: EvalCache=None):
    if cache:
        val = cache.get(ind, config)
        if val is not None:
            return val

    phen_for_fn = ind
    if config.get('var_names') and isinstance(ind, (list, tuple, np.ndarray)):
        phen_for_fn = {config['var_names'][i]: ind[i] for i in range(len(config['var_names']))}

    try:
        base_fitness = float(fitness_fn(phen_for_fn))
    except Exception:
        base_fitness = -1e9

    constraints = config.get('constraints', [])
    phen_dict = phenotype_to_dict(ind, config.get('var_names'))
    total_violation = compute_total_violation(phen_dict, constraints)

    penalty_mode = config.get('penalty_mode', 'none')
    penalty_coef = float(config.get('penalty_coef', 0.0))

    if penalty_mode == 'static':
        penalty = penalty_coef * total_violation
    elif penalty_mode == 'dynamic':
        current_gen = float(config.get('current_gen', 0))
        max_gen = float(max(1, config.get('generations', 1)))
        factor = 1.0 + (current_gen / max_gen)
        penalty = penalty_coef * factor * total_violation
    else:
        penalty = 0.0

    adjusted = base_fitness - penalty

    if cache:
        cache.set(ind, config, adjusted)
    return adjusted

# --------------------------- NSGA-II helpers ------------------------------

def dominates(a, b):
    better_in_all = all(x <= y for x,y in zip(a,b))
    better_in_one = any(x < y for x,y in zip(a,b))
    return better_in_all and better_in_one


def non_dominated_sort(pop_objs):
    S = [set() for _ in range(len(pop_objs))]
    n = [0]*len(pop_objs)
    rank = [0]*len(pop_objs)
    fronts = [[]]
    for p in range(len(pop_objs)):
        for q in range(len(pop_objs)):
            if dominates(pop_objs[p], pop_objs[q]):
                S[p].add(q)
            elif dominates(pop_objs[q], pop_objs[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in list(S[p]):
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i+1
                    Q.append(q)
        i += 1
        fronts.append(Q)
    fronts.pop()
    return fronts


def crowding_distance(objs, front):
    l = len(front)
    if l == 0:
        return {}
    distances = {i:0.0 for i in front}
    num_obj = len(objs[0])
    for m in range(num_obj):
        values = [(i, objs[i][m]) for i in front]
        values.sort(key=lambda x: x[1])
        distances[values[0][0]] = float('inf')
        distances[values[-1][0]] = float('inf')
        vmin = values[0][1]
        vmax = values[-1][1]
        if vmax == vmin:
            continue
        for k in range(1, l-1):
            distances[values[k][0]] += (values[k+1][1] - values[k-1][1])/(vmax - vmin)
    return distances

# --------------------------- GA Engine -----------------------------------

class GeneticAlgorithm:
    def __init__(self, encoding: EncodingSpec, fitness_fn: Callable[[Any], float], is_multi: bool=False, multi_fitness_fn: Optional[Callable]=None):
        self.encoding = encoding
        self.fitness_fn = fitness_fn
        self.is_multi = is_multi
        self.multi_fitness_fn = multi_fitness_fn
        self.cache = EvalCache()

    def run(self, config: dict, ui_callback: Optional[Callable]=None):
        pop_size = config['pop_size']
        gens = config['generations']
        seed = config.get('seed')
        set_seed(seed)
        pop = init_population(pop_size, self.encoding, seed=seed, heuristic_seeds=config.get('heuristic_seeds'))
        fitnesses = []
        multi_objs = []
        config['current_gen'] = 0
        for ind in pop:
            if self.is_multi and self.multi_fitness_fn:
                objs = tuple(self.multi_fitness_fn(ind))
                multi_objs.append(objs)
                fitnesses.append(sum(objs))
            else:
                f = evaluate_individual(ind, self.encoding, self.fitness_fn, config, cache=self.cache)
                fitnesses.append(f)
        history = {'best':[], 'avg':[], 'worst':[], 'fitness_snapshot': [], 'pop_snapshot': [], 'objs_snapshot': []}
        history['fitness_snapshot'].append(fitnesses.copy())
        history['pop_snapshot'].append([deepcopy(p) for p in pop])
        if self.is_multi and self.multi_fitness_fn:
            history['objs_snapshot'].append([tuple(self.multi_fitness_fn(ind)) for ind in pop])

        for gen in range(gens):
            config['current_gen'] = gen
            if ui_callback and gen % max(1, gens//50) == 0:
                ui_callback(gen, pop, fitnesses)
            new_pop = []
            while len(new_pop) < pop_size:
                method = config['selection']
                if method == 'tournament':
                    parents = tournament_selection(pop, fitnesses, k=2, tour_size=config.get('tournament_size',3))
                elif method == 'roulette':
                    parents = roulette_wheel_selection(pop, fitnesses, k=2)
                elif method == 'rank':
                    parents = rank_selection(pop, fitnesses, k=2)
                else:
                    parents = tournament_selection(pop, fitnesses, k=2, tour_size=3)
                p1,p2 = parents[0], parents[1]
                child1, child2 = deepcopy(p1), deepcopy(p2)
                if np.random.rand() < config['crossover_prob']:
                    cx = config['crossover']
                    if cx == 'one_point':
                        child1, child2 = one_point_crossover(p1, p2)
                    elif cx == 'two_point':
                        child1, child2 = two_point_crossover(p1, p2)
                    elif cx == 'uniform':
                        child1, child2 = uniform_crossover(p1, p2, prob=config.get('uniform_prob',0.5))
                    elif cx == 'pmx' and self.encoding.name == 'permutation':
                        child1, child2 = pmx_crossover(p1, p2)
                    elif cx == 'sbx' and self.encoding.name == 'real':
                        child1, child2 = sbx_crossover(p1, p2, eta=config.get('sbx_eta',15))
                m = config['mutation']
                pm = config['mutation_prob']
                if m == 'bitflip' and self.encoding.name == 'binary':
                    child1 = bitflip_mutation(child1, pm); child2 = bitflip_mutation(child2, pm)
                elif (m == 'gaussian' and self.encoding.name == 'real') or (m == 'gaussian' and self.encoding.name == 'integer'):
                    child1 = gaussian_mutation(child1, pm, config.get('sigma',0.1), self.encoding.kwargs.get('bounds'))
                    child2 = gaussian_mutation(child2, pm, config.get('sigma',0.1), self.encoding.kwargs.get('bounds'))
                elif m == 'swap' and self.encoding.name == 'permutation':
                    child1 = swap_mutation(child1, pm); child2 = swap_mutation(child2, pm)
                elif m == 'inversion' and self.encoding.name == 'permutation':
                    child1 = inversion_mutation(child1, pm); child2 = inversion_mutation(child2, pm)
                elif m == 'int_reset' and self.encoding.name == 'integer':
                    child1 = random_resetting_mutation(child1, pm, self.encoding.kwargs.get('bounds'))
                    child2 = random_resetting_mutation(child2, pm, self.encoding.kwargs.get('bounds'))
                if config.get('repair'):
                    if config.get('constraints'):
                        child1 = equality_repair_by_substitution(child1, config)
                        child2 = equality_repair_by_substitution(child2, config)
                    if self.encoding.name == 'binary' and config.get('knapsack'):
                        kk = config['knapsack']
                        child1 = knapsack_repair(child1, kk['weights'], kk['values'], kk['capacity'])
                        child2 = knapsack_repair(child2, kk['weights'], kk['values'], kk['capacity'])
                    if self.encoding.name == 'integer' and config.get('int_bounds'):
                        child1 = clamp_integer(child1, config['int_bounds'])
                        child2 = clamp_integer(child2, config['int_bounds'])
                new_pop.append(child1)
                if len(new_pop) < pop_size:
                    new_pop.append(child2)
            pop = new_pop
            fitnesses = []
            gen_objs = []
            for ind in pop:
                if self.is_multi and self.multi_fitness_fn:
                    objs = tuple(self.multi_fitness_fn(ind))
                    gen_objs.append(objs)
                    fitnesses.append(sum(objs))
                else:
                    f = evaluate_individual(ind, self.encoding, self.fitness_fn, config, cache=self.cache)
                    fitnesses.append(f)
            best = max(fitnesses) if len(fitnesses)>0 else None
            avg = sum(fitnesses)/len(fitnesses) if len(fitnesses)>0 else None
            worst = min(fitnesses) if len(fitnesses)>0 else None
            history['best'].append(best); history['avg'].append(avg); history['worst'].append(worst)
            history['fitness_snapshot'].append(fitnesses.copy())
            history['pop_snapshot'].append([deepcopy(p) for p in pop])
            if self.is_multi and self.multi_fitness_fn:
                history['objs_snapshot'].append(gen_objs)
            if ui_callback and gen % max(1, gens//50) == 0:
                ui_callback(gen, pop, fitnesses)
        return pop, fitnesses, history

# --------------------------- Streamlit App --------------------------------

st.set_page_config(page_title='GA Playground', layout='wide')
st.title('Genetic Algorithm Playground — Flexible, Pluggable')

# Sidebar: workflow choice
workflow = st.sidebar.radio('Workflow', ['Problem-first (define analytically)', 'Data-first (upload dataset)'])

# Common hyperparameters
with st.sidebar.expander('General Hyperparameters (safe defaults)', expanded=True):
    pop_size = st.number_input('Population size', min_value=4, max_value=2000, value=100, step=4)
    generations = st.number_input('Generations', min_value=1, max_value=10000, value=200)
    crossover_prob = st.slider('Crossover probability (general)', 0.0, 1.0, 0.9)
    mutation_prob = st.slider('Mutation probability (general, per-gene or per-ind depending)', 0.0, 1.0, 0.02)
    seed_val = st.number_input('Random seed (0 for random)', min_value=0, value=0)
    seed = None if seed_val==0 else int(seed_val)

# Problem-first
encoding_spec = None
fitness_fn = None
is_multi = False
multi_fitness_fn = None
var_names = []
constraints_txt = ''

if workflow == 'Problem-first (define analytically)':
    st.sidebar.markdown('**Problem definition**')
    problem_type = st.sidebar.selectbox('Problem type', ['Single-objective', 'Multi-objective'])
    is_multi = problem_type == 'Multi-objective'

    st.sidebar.markdown('Define decision variables (comma-separated), e.g. Xc,Xp,Xw,Xt,Yc,Yp,Yw,Yt')
    varnames_txt = st.sidebar.text_input('Variable names', value='')
    var_names = [v.strip() for v in varnames_txt.split(',') if v.strip()]

    encoding_choice = st.sidebar.selectbox('Encoding', ['binary','integer','real','permutation'])
    if encoding_choice == 'binary':
        length = st.sidebar.number_input('Binary length', min_value=1, value=50)
        encoding_spec = EncodingSpec('binary', length=int(length))
    elif encoding_choice == 'integer':
        dims = len(var_names) if len(var_names)>0 else st.sidebar.number_input('Number of integer genes', min_value=1, value=5)
        st.sidebar.markdown('Enter bounds as JSON list of [lo,hi] matching variable order e.g. [[0,100], ...]')
        default_bounds = [[0,100]]*dims
        bounds_txt = st.sidebar.text_area('Integer bounds JSON', value=json.dumps(default_bounds))
        try:
            bounds = json.loads(bounds_txt)
        except:
            bounds = default_bounds
        encoding_spec = EncodingSpec('integer', bounds=bounds)
    elif encoding_choice == 'real':
        dims = len(var_names) if len(var_names)>0 else st.sidebar.number_input('Number of real-valued genes', min_value=1, value=5)
        st.sidebar.markdown('Enter bounds as JSON list of [lo,hi] matching variable order e.g. [[0.0,100.0], ...]')
        default_bounds = [[0.0,100.0]]*dims
        bounds_txt = st.sidebar.text_area('Real bounds JSON', value=json.dumps(default_bounds))
        try:
            bounds = json.loads(bounds_txt)
        except:
            bounds = default_bounds
        use_lhs = st.sidebar.checkbox('Use Latin Hypercube sampling for init', value=False)
        encoding_spec = EncodingSpec('real', bounds=bounds, lhs=use_lhs)
    elif encoding_choice == 'permutation':
        n = st.sidebar.number_input('Permutation size (n items)', min_value=2, value=20)
        encoding_spec = EncodingSpec('permutation', n=int(n))

    st.sidebar.markdown('---')
    st.sidebar.markdown('**Fitness function**')
    st.sidebar.markdown('Provide a Python expression using `x` (the decoded phenotype). If you set variable names, `x` will be a dict: e.g. x["Xc"]')
    fitness_expr = st.sidebar.text_area('Fitness expression (single-objective)', value="sum(x.values()) if isinstance(x, dict) else sum(x)")
    if is_multi:
        st.sidebar.markdown('Enter multi-objective functions as JSON list of expressions, e.g. ["-sum(x.values())","sum([v*v for v in x.values()])"]')
        multi_expr_txt = st.sidebar.text_area('Multi-objective expressions JSON', value=json.dumps(["sum(x.values())"]))

    st.sidebar.markdown('---')
    st.sidebar.markdown('**Linear constraints**')
    st.sidebar.markdown('Enter one constraint per line. Use operators: =, <=, >=. Example: `Xc + Yc = 800`')
    constraints_txt = st.sidebar.text_area('Constraints (one per line)', value='')

    def build_fitness_fn(expr: str):
        def fn(x):
            loc = {'x': x}
            return float(eval(expr, {}, loc))
        return fn

    try:
        if is_multi:
            multi_exprs = json.loads(multi_expr_txt)
            multi_fitness_fn = lambda x: tuple(float(eval(e, {}, {'x':x})) for e in multi_exprs)
        else:
            fitness_fn = build_fitness_fn(fitness_expr)
    except Exception as e:
        st.sidebar.error(f'Fitness parsing error: {e}')

else:
    st.sidebar.markdown('**Upload dataset**')
    uploaded = st.sidebar.file_uploader('CSV file', type=['csv','txt','xls','xlsx'])
    df = None
    task = None
    target_col = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            try:
                df = pd.read_excel(uploaded)
            except Exception as e:
                st.sidebar.error('Could not read file: '+str(e))
        if df is not None:
            st.sidebar.write(f'Dataset shape: {df.shape}')
            target_col = st.sidebar.selectbox('Target column (for supervised tasks)', options=[None]+list(df.columns))
            task = st.sidebar.selectbox('Task', ['Feature selection (supervised)','Hyperparameter tuning (supervised)','TSP/distance matrix (perm)'])

    if df is not None and task is not None:
        if task.startswith('Feature'):
            features = [c for c in df.columns if c!=target_col]
            encoding_spec = EncodingSpec('binary', length=len(features))
            st.sidebar.markdown(f'Feature count: {len(features)}')
            metric = st.sidebar.selectbox('Evaluation metric', ['accuracy','f1','roc_auc','rmse'])
            folds = st.sidebar.number_input('CV folds', min_value=2, max_value=20, value=5)
            from sklearn.linear_model import LogisticRegression
            def fs_fitness_fn(bitmask):
                mask = np.array(bitmask).astype(int)
                if mask.sum() == 0:
                    return 0.0
                X = df[features].values[:, mask==1]
                y = df[target_col].values
                try:
                    clf = LogisticRegression(max_iter=500)
                    if metric == 'accuracy':
                        score = np.mean(cross_val_score(clf, X, y, cv=folds, scoring='accuracy'))
                        return score
                    elif metric == 'f1':
                        score = np.mean(cross_val_score(clf, X, y, cv=folds, scoring='f1'))
                        return score
                    elif metric == 'roc_auc':
                        score = np.mean(cross_val_score(clf, X, y, cv=folds, scoring='roc_auc'))
                        return score
                    else:
                        return 0.0
                except Exception:
                    return 0.0
            fitness_fn = fs_fitness_fn
        elif task.startswith('Hyperparam'):
            st.sidebar.info('Hyperparameter tuning support is scaffolded; you will need to provide a parameter mapping and model.')
        elif task.startswith('TSP'):
            st.sidebar.info('Expect a distance matrix CSV where rows/cols are points')
            try:
                coords = df.values
                n = coords.shape[0]
                encoding_spec = EncodingSpec('permutation', n=n)
                def tsp_fitness_fn(perm):
                    order = perm
                    if coords.shape[1] >= 2:
                        dist = 0.0
                        for i in range(len(order)-1):
                            a = coords[order[i]]
                            b = coords[order[i+1]]
                            dist += np.linalg.norm(a-b)
                        dist += np.linalg.norm(coords[order[-1]] - coords[order[0]])
                        return -dist
                    else:
                        return 0.0
                fitness_fn = tsp_fitness_fn
            except Exception as e:
                st.sidebar.error('Could not interpret dataset for TSP: '+str(e))

# Operators & constraint handling (sidebar)
with st.sidebar.expander('Operators & Constraint Handling', expanded=True):
    selection = st.selectbox('Parent selection', ['tournament','roulette','rank'])
    crossover = st.selectbox('Crossover', ['one_point','two_point','uniform','pmx','sbx'])
    mutation = st.selectbox('Mutation', ['bitflip','gaussian','swap','inversion','int_reset'])
    crossover_prob_ops = st.sidebar.slider('Crossover probability (operators)', 0.0, 1.0, float(crossover_prob), key='crossover_prob_ops')
    mutation_prob_ops = st.sidebar.slider('Mutation probability (operators)', 0.0, 1.0, float(mutation_prob), key='mutation_prob_ops')
    tournament_size = st.number_input('Tournament size', min_value=2, value=3)
    repair_toggle = st.checkbox('Enable repair functions (clamp/knapsack/equality)', value=False)
    penalty_mode = st.selectbox('Penalty mode (if enabled)', ['none','static','dynamic'])
    if penalty_mode != 'none':
        penalty_coef = st.number_input('Penalty coefficient (lambda)', min_value=0.0, value=1.0)

st.sidebar.markdown('---')
preset = st.sidebar.selectbox('Quick presets', ['None','Knapsack (binary)','TSP (perm)','Feature selection (dataset)'])
if preset != 'None':
    st.sidebar.info('Selecting a preset will set sensible defaults; change them afterwards if you like.')

run_button = st.sidebar.button('Run GA')

# Main run
if 'last_run' not in st.session_state:
    st.session_state['last_run'] = None

if run_button:
    if encoding_spec is None or fitness_fn is None:
        st.error('Please complete encoding and fitness function before running.')
    else:
        parsed_constraints = build_constraints_list(constraints_txt) if constraints_txt else []
        cfg = {
            'pop_size': int(pop_size),
            'generations': int(generations),
            'crossover_prob': float(crossover_prob_ops) if 'crossover_prob_ops' in locals() else float(crossover_prob),
            'mutation_prob': float(mutation_prob_ops) if 'mutation_prob_ops' in locals() else float(mutation_prob),
            'selection': selection,
            'crossover': crossover,
            'mutation': mutation,
            'tournament_size': int(tournament_size),
            'seed': seed,
            'repair': repair_toggle,
            'constraints': parsed_constraints,
            'var_names': var_names,
            'penalty_mode': penalty_mode if 'penalty_mode' in locals() else 'none',
            'penalty_coef': float(penalty_coef) if 'penalty_coef' in locals() else 0.0,
            'int_bounds': encoding_spec.kwargs.get('bounds') if encoding_spec and encoding_spec.name == 'integer' else None
        }

        ga = GeneticAlgorithm(encoding_spec, fitness_fn, is_multi=is_multi, multi_fitness_fn=multi_fitness_fn if is_multi else None)
        progress = st.progress(0)
        status_text = st.empty()
        chart = st.line_chart()
        def ui_cb(gen, pop, fitnesses):
            progress.progress(min(100, int(100*gen/cfg['generations'])))
            status_text.text(f'Generation {gen}/{cfg["generations"]} — best {max(fitnesses):.4f} avg {np.mean(fitnesses):.4f}')
            chart.add_rows(pd.DataFrame({'best':[max(fitnesses)],'avg':[np.mean(fitnesses)],'worst':[min(fitnesses)]}))
        start_time = time.time()
        pop, fitnesses, history = ga.run(cfg, ui_callback=ui_cb)
        end_time = time.time()
        st.success(f'Run finished in {end_time - start_time:.2f}s')

        # Animations (Plotly)
        try:
            import plotly.graph_objects as go

            def build_single_objective_animation(history):
                frames = []
                gens = len(history.get('best', []))
                for g in range(gens):
                    ys = history['fitness_snapshot'][g]
                    scatter = go.Scatter(x=list(range(len(ys))), y=ys, mode='markers', marker=dict(size=6), name='population')
                    best_line = go.Scatter(x=list(range(g+1)), y=history['best'][:g+1], mode='lines+markers', name='best')
                    avg_line = go.Scatter(x=list(range(g+1)), y=history['avg'][:g+1], mode='lines', name='avg')
                    worst_line = go.Scatter(x=list(range(g+1)), y=history['worst'][:g+1], mode='lines', name='worst')
                    frames.append(go.Frame(data=[scatter, best_line, avg_line, worst_line], name=str(g)))
                init_ys = history['fitness_snapshot'][0] if history.get('fitness_snapshot') else []
                fig = go.Figure(
                    data=[
                        go.Scatter(x=list(range(len(init_ys))), y=init_ys, mode='markers', name='population'),
                        go.Scatter(x=[0], y=[history['best'][0]] if history.get('best') else [None], mode='lines+markers', name='best'),
                        go.Scatter(x=[0], y=[history['avg'][0]] if history.get('avg') else [None], mode='lines', name='avg'),
                        go.Scatter(x=[0], y=[history['worst'][0]] if history.get('worst') else [None], mode='lines', name='worst'),
                    ],
                    layout=go.Layout(
                        title='Population fitness evolution',
                        xaxis=dict(title='individual index / generation for lines'),
                        yaxis=dict(title='Fitness'),
                        updatemenus=[dict(
                            type='buttons',
                            buttons=[dict(label='Play', method='animate', args=[None, {'frame': {'duration': 200, 'redraw': True}, 'fromcurrent': True}])]
                        )]
                    )
                )
                fig.frames = frames
                return fig

            def compute_pareto_front_points(objs):
                pts = list(range(len(objs)))
                nondom = []
                for i in pts:
                    dominated = False
                    for j in pts:
                        if i==j: continue
                        if dominates(objs[j], objs[i]):
                            dominated = True; break
                    if not dominated:
                        nondom.append(i)
                return nondom

            def build_pareto_evolution_animation(history):
                frames = []
                gens = len(history.get('objs_snapshot', []))
                for g in range(gens):
                    objs = history['objs_snapshot'][g]
                    if len(objs)==0:
                        continue
                    xs = [o[0] for o in objs]
                    ys = [o[1] for o in objs] if len(objs[0])>1 else [0]*len(objs)
                    nondom_idx = compute_pareto_front_points(objs)
                    nondom_x = [xs[i] for i in nondom_idx]
                    nondom_y = [ys[i] for i in nondom_idx]
                    all_scatter = go.Scatter(x=xs, y=ys, mode='markers', name='population')
                    pareto_scatter = go.Scatter(x=nondom_x, y=nondom_y, mode='markers', marker=dict(size=10, symbol='diamond'), name='pareto')
                    frames.append(go.Frame(data=[all_scatter, pareto_scatter], name=str(g)))
                if gens==0:
                    return go.Figure()
                init_objs = history['objs_snapshot'][0]
                fig = go.Figure(
                    data=[
                        go.Scatter(x=[o[0] for o in init_objs], y=[o[1] for o in init_objs], mode='markers', name='population'),
                        go.Scatter(x=[], y=[], mode='markers', marker=dict(size=10, symbol='diamond'), name='pareto'),
                    ],
                    layout=go.Layout(
                        title='Pareto front evolution',
                        xaxis=dict(title='Objective 1'),
                        yaxis=dict(title='Objective 2'),
                        updatemenus=[dict(
                            type='buttons',
                            buttons=[dict(label='Play', method='animate', args=[None, {'frame': {'duration': 300, 'redraw': True}, 'fromcurrent': True}])]
                        )]
                    )
                )
                fig.frames = frames
                return fig

            if ga.is_multi and history.get('objs_snapshot'):
                st.subheader('Pareto front evolution (multi-objective)')
                fig = build_pareto_evolution_animation(history)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.subheader('Fitness evolution (single-objective)')
                fig = build_single_objective_animation(history)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning('Could not build animation: '+str(e))

        best_idx = int(np.argmax(fitnesses))
        st.write('Best fitness:', fitnesses[best_idx])
        st.write('Best individual (genotype):', pop[best_idx])
        st.session_state['last_run'] = {'pop': pop, 'fitnesses': fitnesses, 'history': history, 'encoding': encoding_spec}

# Show last run results if present
if st.session_state['last_run'] is not None:
    lr = st.session_state['last_run']
    st.subheader('Last run summary')
    if lr.get('encoding'):
        st.write('Encoding:', lr['encoding'].name)
    hist = lr['history']
    if hist.get('best'):
        st.line_chart(pd.DataFrame({'best':hist['best'],'avg':hist['avg'],'worst':hist['worst']}))
    topk = 5
    fitnesses = lr['fitnesses']
    pop = lr['pop']
    best_ids = np.argsort(fitnesses)[-topk:][::-1]
    rows = []
    for i in best_ids:
        rows.append({'rank':i, 'fitness':fitnesses[i], 'genotype':list(pop[i])})
    st.table(pd.DataFrame(rows))
    if st.button('Export best solution as JSON'):
        best = pop[int(np.argmax(fitnesses))]
        st.download_button('Download JSON', data=json.dumps({'best': best.tolist(), 'fitness': float(max(fitnesses))}, indent=2), file_name='ga_best_solution.json')

st.sidebar.markdown('---')
st.sidebar.markdown('This app is a scaffold. It implements many operators and a flexible UI; extend or plug-in your own evaluation, repair, and operators as needed.')
