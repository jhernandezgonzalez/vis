#!/usr/bin/env python
# coding: utf-8


import numpy as np
import math 
import itertools as it
import scipy.stats as sc
from scipy.special import logsumexp

#from FrechetBound import discrete_frechet_lower_bound, discrete_frechet_upper_bound, define_h_lower_based, define_h_upper_based, define_h_both

def nat_increase_entropy(beta, entropy_multiplier):
    new_beta = beta.copy()
    new_beta *= entropy_multiplier
    new_beta -= logsumexp(new_beta)
    return new_beta

def to_mean(beta):
    max_beta = np.max(beta)
    p_hat = np.exp(beta - max_beta)
    p = p_hat / np.sum(p_hat)
    return p

def to_nat(p):
    return np.log(p)  
  
# Parameter generators in natural parametrization

def normal_params(stdev):
    return lambda n : np.random.normal(loc = 0 , scale=stdev, size=n)

def bounded_uniform(scale):
    return lambda n: scale * np.random.random(n)

# Parameter generators in mean parametrization

def flat_dirichlet(alpha):
    return lambda n: np.random.dirichlet(np.ones(n)*alpha)    

# Transformations

def mean_gen(_nat_gen):
    return lambda n: to_mean(_nat_gen(n))

def nat_gen(_mean_gen):
    return lambda n: to_nat(_mean_gen(n))

# Cartesian product distributions

def max_entropy_from_marginals(cardinalities, margs):
    #dim = np.product(cardinalities)
    nat_p = np.zeros(cardinalities)
    for i, m in enumerate(margs):
        shape = np.ones(len(cardinalities), dtype=int)
        shape[i] = cardinalities[i]
        old_shape = m.shape
        m.shape = shape
        nat_p += m
        m.shape = old_shape
    return nat_p

def max_entropy_from_marginals_but_h(cardinalities, margs, h):
    #dim = np.product(cardinalities)
    new_cards = np.delete(cardinalities,h)
    nat_p = np.zeros(new_cards)
    one_less = 0
    for i, m in enumerate(margs):
        if i == h:
            one_less = 1
        else:
            shape = np.ones(len(new_cards), dtype=int)
            shape[i-one_less] = cardinalities[i]
            old_shape = m.shape
            m.shape = shape
            nat_p += m
            m.shape = old_shape
    return nat_p
    
def gen_max_entropy_from_marginals(cardinalities, nat_gen):
  margs = []
  for card in cardinalities:
    m = nat_gen(card)
    #m.shape=(1, card)
    margs.append(m)
    #print(to_mean(m))
  return max_entropy_from_marginals(cardinalities, margs)

def to_mean_cp(nat_cp):
    s = nat_cp.shape
    f_nat_cp = nat_cp.flatten()
    mean_cp = to_mean(f_nat_cp)
    mean_cp.shape = s
    return mean_cp
    
def to_nat_cp(mean_cp):
    s = mean_cp.shape
    f_mean_cp = mean_cp.flatten()
    nat_cp = to_nat(f_mean_cp)
    nat_cp.shape = s
    return nat_cp

def nat_gen_cp(nat_gen):
    def gen_cp(cards):
        #print(cards)
        k = np.product(cards)
        #print(k)
        p = nat_gen(k)
        p.shape = cards
        return p
    return gen_cp
    
def mixture_problem(cards, lambda_mix, f_marg_nat_gen, f_dep_nat_gen, p_nat_gen_cp):
    f_ind = to_mean_cp(gen_max_entropy_from_marginals(cards, f_marg_nat_gen))
    f_dep = to_mean(f_dep_nat_gen(np.prod(cards)))
    f_dep.shape = cards
    f = f_ind * lambda_mix + f_dep * (1. - lambda_mix)
    
    p = to_mean_cp(p_nat_gen_cp(cards))
    
    #fp/=np.sum(fp)
    #print(np.max(fp), np.min(fp), (fp).flatten()[:200])
    
    return f, p


def dependent_naive_problem(cards, f_marg_nat_gen, p_nat_gen_cp, card_c, c_marg_nat_gen, gamma_mix=0):
    lMargs = []
    for card in cards:
        m = to_mean(f_marg_nat_gen(card))
        lMargs.append(m)

    nat_pdf_c = c_marg_nat_gen(card_c)
    pdf_c = to_mean(nat_pdf_c)

    lConditionals = []
    for pdf_x in lMargs:
        nat_pdf_h = to_nat_cp( define_h_both(pdf_x, pdf_c, gamma_mix) )
        nat_pdf_h -= nat_pdf_c[np.newaxis, :]

        lConditionals.append(nat_pdf_h)#

    f = np.zeros(cards)
    for c in np.arange(len(pdf_c)):
        lMargsfc = []
        for v in np.arange(len(cards)):
            lMargsfc.append(lConditionals[v][:, c])
        f_c = to_mean_cp(max_entropy_from_marginals(cards, lMargsfc))

        f += f_c * pdf_c[c]

    # for m in np.arange(len(cards)):
    #     print(np.sum(f,axis=tuple(np.delete(np.arange(len(cards)),m))))
    # print(np.sum(f),np.sort(f.flatten())[-12:], np.sort(f.flatten())[:12])

    p = to_mean_cp(p_nat_gen_cp(cards))

    #fp = f * p
    #fp /= np.sum(fp)
    # print(np.sum(fp),np.sort(fp.flatten())[-12:], np.sort(fp.flatten())[:12])

    return f, p
