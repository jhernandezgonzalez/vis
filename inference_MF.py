import numpy as np
from scipy.special import logsumexp
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression
import resource

from pyGM.graphmodel import GraphModel
from pyGM.factor import Factor
from pyGM.varset_py import VarSet


def cpu_time():
    return resource.getrusage(resource.RUSAGE_SELF)[0]

''' TODO : General Guides
- neighbors 1 hop
- neighbors 2 hop
- Minimum entropy factors
- error de aproximación en funcion de q


- Revisar normalizacion: introducir 1/d(X) para compensar la eliminación de variables?
'''


# Recall that:
#   r=1 -> M-projection
#   r=1e-2 -> Almost I-projection
#   r=2 -> Best for importance sampling
default_r = 2.0
default_tol = 1e-17
default_num_it = 100
default_step = 0.5
default_limit_nvars = 7

#################################################################################################
####################################### Auxiliar functions ######################################
#################################################################################################

def domains_k(p: GraphModel, k):
    X = p.X
    if k >= len(X):
        return [X]
    else:
        return [VarSet(X[i:i+k]) for i in range(len(X)-k+1)]


def _copies_to_marginals(copies, vars_to_facs, cardinalities, normalize=True):
    marginals = [None]*len(cardinalities)

    for ind_v, ind_facs in vars_to_facs.items():
        nat_qi = np.zeros(cardinalities[ind_v])
        for ind_f in ind_facs:
            nat_qi += copies[ind_f][ind_v]

        if normalize:
            nat_qi -= logsumexp(nat_qi)

        marginals[ind_v] = nat_qi

    return marginals


def _factor_maker(d, isLog=True, normalized=True):
    f = Factor(d)
    if normalized:
        if isLog:
            f.normalizeLogIP()
        else:
            f.normalizeIP()
    return f


def _void_gm(domains, isLog=True, normalized=True):
    factors = []
    for d in domains:
        f = _factor_maker(d, isLog=isLog, normalized=normalized)
        factors.append(f)
    return GraphModel(factors, isLog=isLog)


def _copy_and_void_gm(p: GraphModel, isLog=True, normalized=True):
    factors = []
    for orig_f in p.factors:
        f = _factor_maker(orig_f.v, isLog=isLog, normalized=normalized)
        factors.append(f)
    return GraphModel(factors, isLog=isLog)


def _convergence_list(old_q:list, q:list, tol:float):
    for old_qi, new_qi in zip(old_q, q):
        if not np.allclose(old_qi, new_qi, atol=tol):
            return False
    return True


#################################################################################################
##################################### Projection functions ######################################
#################################################################################################

class Projection:
    """
    General abstract class for projection methods
    """
    def __init__(self, p, family=None):
        """
        :param p: function to project
        :param family: family to project onto:
           it might be an integer (size of the factors of q),
           the list of domains of the factors of q, or
           or directly a PGM
        """
        if isinstance(family, GraphModel):
            self.phis = family
        else:
            if family is None:
                family = domains_k(p, 1)
            elif isinstance(family, int):
                family = domains_k(p, family)
            self.phis = _void_gm(family)
        self.psis = p.copy()
        self.psis.toLog() # Here we work in logarithmic space
        self.phis.toLog() # Here we work in logarithmic space

    def set_target_q(self, phis):
        self.phis = phis
        self.phis.toLog() # Here we work in logarithmic space

    def q(self):
        return self.phis

    def _update(self):
        raise Exception("Not implemented")

    def _initialize(self):
        raise Exception("Not implemented")

    def _convergence(self, old_q: GraphModel, q: GraphModel):
        for old_qi, new_qi in zip(old_q.factors, q.factors):
            if not np.allclose(old_qi.t, new_qi.t, atol=self.tol):
                return False
        return True

    def project(self, step = default_step, tol=default_tol, num_it=default_num_it,
                verbose=False, normalize_qs=True):
        self.step = step
        self.tol = tol
        self.num_it = num_it
        self.verbose = verbose
        self.normalize_qs = normalize_qs

        if self.verbose: print(" **** Project", self,"**** ")


        #start = cpu_time()
        self._initialize()
        #end = cpu_time()
        #print("[time] init:", end - start)

        q = self.q()#.copy()
        if self.verbose: print("It( 0 ):", [phi.normalizeLog().t for phi in self.phis.factors])

        it = 0
        self.convergence = False

        #start = cpu_time()

        while (it < num_it) and (not self.convergence):
            old_q = q.copy()
            q = self._update()

            it += 1
            self.convergence = self._convergence(old_q, q)

            if self.verbose:
                #print("It(", it, "):", [phi.normalizeLog().t for phi in self.phis.factors])
                print("It(", it, "):", self.psis.joint().expIP().normalizeIP().distance(self.phis.joint().expIP().normalizeIP(),"r2"))
        #end = cpu_time()
        #print("[time] altra:", end - start)

        if self.normalize_qs:
            for phi in self.phis.factors:
                phi.normalizeLogIP()

        return q

    def is_converged(self):
        return self.convergence

"""
 R-projection Minka's inpired [VIS-Stuff, Overleaf] (Jesus' implementation)

 Monolithic R-projection functions
 - p should be a factor which contains the logs of the values.
 - q should be a gm in log with every factor normalized.
"""
class MonolithicRProjection(Projection):
    def __init__(self, p: GraphModel, family=None, r=default_r):
        Projection.__init__(self, p, family)
        self.r = r

    def _initialize(self):
        self.p_pow_r = self.psis.factors[0].copy()
        self.p_pow_r *= self.r # assuming natural params
        self.q_joint = self.q().joint()

    def _update_phi(self, phi:Factor):
        self.q_joint -= phi
        phi.table = ((1. / self.r) * ((self.p_pow_r + (1-self.r) * self.q_joint).lsemarginal(phi.vars))).table
        #phi.normalizeLogIP() # TODO: hace falta normalizar aquí?
        #todo: falta aqui un logarithm?
        self.q_joint += phi

    def _update(self):
        for phi in self.q().factors:
            self._update_phi(phi)
        return self.phis


class FactoredRProjection(Projection):
    """
    Factored r-projection class
    """
    def __init__(self, p, family=None, r=default_r):
        Projection.__init__(self, p, family)
        self.r = r

    def _initialize(self):
        self.xi = {}
        self.xi_t = {}
        self.phi_gms = {}
        self.psi_gms = {}
        for psi in self.psis.factors:
            self.xi[psi] = {}
            for phi in self.phis.factors:
                # Now we only take those phis whose domain is a subset of the domain of psi
                # We could alternatively create a factor with the intersection of the domains whenever the factors overlap.
                # Indeed, I think that will be better.
                if phi.vars.issubset(psi.vars):
                    if phi not in self.xi_t:
                        self.xi_t[phi] = {}
                    self.xi[psi][phi] = _factor_maker(phi.vars)
                    self.xi_t[phi][psi] = self.xi[psi][phi]
        for phi in self.phis.factors:
            self.phi_gms[phi] = GraphModel(self.xi_t[phi].values(), copy=False, isLog=True)
        for psi in self.psis.factors:
            self.psi_gms[psi] = GraphModel(self.xi[psi].values(), copy=False, isLog=True)
        self._compute_phis_from_xi()

    def _compute_phis_from_xi(self):
        for phi in self.phis.factors:
            phi.table = self.phi_gms[phi].joint().table

    def _update(self):
        for psi in self.psis.factors:
            #print("Projecting ", psi, psi.table)
            self._project_psi(psi)
        #self._compute_phis_from_xi()
        return self.phis

    def _xi_marginal(self, psi): # In this version, only psi-related phi factors are considered
        f = Factor(vals=0.0)
        #print("+++++++++++++++++++++++++++++++++++++++")
        for phi in self.xi[psi]:
            # Notice that we modify phis
            #s = np.log(phi.exp().sum()) # to normalize phi's after product of copies
            #print("remove", self.xi[psi][phi], self.xi[psi][phi].exp().t)
            #print("de", phi.exp().t)
            phi -= self.xi[psi][phi]
            f += phi#.normalizeLog()

        #print("+++++++++++++++++++END++++++++++++++++++++")
        return f.lsemarginal(psi.vars)


    def _xi_marginal_X(self, psi): # In this version, all phi factors are considered
        f = Factor(vals=0.0)
        #print("+++++++++++++++++++++++++++++++++++++++")
        for phi in self.phis.factors:
            if phi in self.xi[psi]:
                # Notice that we modify phis
                #s = np.log(phi.exp().sum()) # to normalize phi's after product of copies
                #print("remove", self.xi[psi][phi], self.xi[psi][phi].t)
                #print("de", phi.t)
                phi -= self.xi[psi][phi]
            f += phi#-s

        #print("+++++++++++++++++++END++++++++++++++++++++")
        return f.lsemarginal(psi.vars)

    '''def _project_psi(self, psi): # In this version, no step is used for updates
        f_to_proj = psi.copy()
        f_to_proj += (1. / self.r) * self._xi_marginal(psi)
        monolithic_gm = GraphModel([f_to_proj], isLog=True)
        proj_family = self.psi_gms[psi]                    # PROJ #1
        #proj_family = _copy_and_void_gm(self.psi_gms[psi]) # PROJ #2
        mono_r_proj = MonolithicRProjection(monolithic_gm, family=proj_family, r=self.r)
        new_psi_gms = mono_r_proj.project(self.step, self.tol, self.num_it, verbose=False, normalize_qs=True)
        for phi in self.xi[psi]:
            # Next 2 lines are required if PROJ #2 is used
            #copia = new_psi_gms.factorsWithAll(phi.vars)[0]
            #self.xi[psi][phi].table = copia.table
            #  Phis modified in _xi_marginal are fixed here
            phi += self.xi[psi][phi]'''

    def _project_psi(self, psi): # In this version, step is used for updates
        f_to_proj = psi.copy()
        #print("asi cambia")
        #print("a",f_to_proj.exp().t)
        f_to_proj += (1. / self.r) * self._xi_marginal(psi)
        #print("b",f_to_proj.exp().t)
        monolithic_gm = GraphModel([f_to_proj], isLog=True)
        proj_family = self.psi_gms[psi].copy()             # PROJ #1
        #proj_family = _copy_and_void_gm(self.psi_gms[psi]) # PROJ #2
        mono_r_proj = MonolithicRProjection(monolithic_gm, family=proj_family, r=self.r)
        new_psi_gms = mono_r_proj.project(self.step, self.tol, self.num_it, verbose=False, normalize_qs=False)
        for phi in self.xi[psi]:
            copia = new_psi_gms.factorsWithAll(phi.vars)[0]
            self.xi[psi][phi].table = self.step * self.xi[psi][phi].table + (1 - self.step) * copia.table
            #  Phis modified in _xi_marginal are fixed here
            #print("add", self.xi[psi][phi], self.xi[psi][phi].exp().t)
            phi += self.xi[psi][phi]
            #print("y queda", phi.exp().t)

    @classmethod
    def straight_project(cls, p:GraphModel, family=None, r=default_r, **kwargs):
                         #step = default_step, tol=default_tol, num_it=default_num_it,
                         #verbose=False, normalize_qs=True):
        proj_method = cls(p, family, r)
        q_model = proj_method.project(**kwargs)
        q_model.toExp()
        return q_model, proj_method.is_converged()



"""
 R-projection Minka's inpired, single q projection
"""
class FactoredRProjectionV2(Projection):
    """
    Factored r-projection class
    """
    def __init__(self, p, family=None, r=default_r):
        Projection.__init__(self, p, family)
        self.r = r

    def _initialize(self):
        self.xi = {}
        self.xi_t = {}
        self.phi_gms = {}
        self.psi_gms = {}
        for psi in self.psis.factors:
            self.xi[psi] = {}
            for phi in self.phis.factors:
                # Now we only take those phis whose domain is a subset of the domain of psi
                # We could alternatively create a factor with the intersection of the domains whenever the factors overlap.
                # Indeed, I think that will be better.
                if phi.vars.issubset(psi.vars):
                    if phi not in self.xi_t:
                        self.xi_t[phi] = {}
                    self.xi[psi][phi] = _factor_maker(phi.vars)
                    self.xi_t[phi][psi] = self.xi[psi][phi]
        for phi in self.phis.factors:
            self.phi_gms[phi] = GraphModel(self.xi_t[phi].values(), copy=False, isLog=True)
        for psi in self.psis.factors:
            self.psi_gms[psi] = GraphModel(self.xi[psi].values(), copy=False, isLog=True)
        self._compute_phis_from_xi()

    def _compute_phis_from_xi(self):
        self.norm_cst = {}
        for phi in self.phis.factors:
            phi.table = self.phi_gms[phi].joint().table
            self.norm_cst[phi] = logsumexp(phi.t)
            phi.table = (phi - self.norm_cst[phi]).table

    def _update(self):
        for psi in self.psis.factors:
            #print("Projecting ", psi, psi.table)
            self._project_psi(psi)
        #self._compute_phis_from_xi()
        return self.phis


    def _project_psi(self, psi): # In this version, step is used for updates
        for phi in self.xi[psi]:
            f_to_proj = psi.copy()
            f_to_proj *= self.r
            for phi_l in self.xi[psi]:
                if phi != phi_l:
                    f_to_proj += (1 - self.r) * phi_l
                    f_to_proj += self.r * (phi_l-self.xi[psi][phi_l])
            f_to_proj += self.r * (phi-self.xi[psi][phi])

            q_a = (1. / self.r) * f_to_proj.lsemarginal(phi.vars)
            q_a.normalizeLogIP()
            #q_a += self.norm_cst[phi]
            q_b = phi - self.xi[psi][phi]
            q_a -= q_b

            # update phi and self.norm_cst[phi]
            phi.table = (phi + self.norm_cst[phi]).table
            phi.table = (phi - self.xi[psi][phi]).table
            #print("a",self.xi[psi][phi],self.xi[psi][phi].table)
            #print("b",q_a,q_a.table)
            self.xi[psi][phi].table = self.step * self.xi[psi][phi].table + (1 - self.step) * q_a.table#q_a.table#
            phi.table = (phi + self.xi[psi][phi]).table
            self.norm_cst[phi] = logsumexp(phi.t)
            phi.table = (phi - self.norm_cst[phi]).table


    @classmethod
    def straight_project(cls, p: GraphModel, family=None, r=default_r, **kwargs):
                         #step = default_step, tol=default_tol, num_it=default_num_it,
                         #verbose=False, normalize_qs=True):
        proj_method = cls(p, family, r)
        q_model = proj_method.project(**kwargs)
        q_model.toExp()
        return q_model, proj_method.is_converged()



"""
Fully Factored (MF) Alpha-projection (Minka 4.1)
"""
class AlphaProjection(Projection):
    def __init__(self, p, family, alpha=default_r):
        Projection.__init__(self, p, family)
        self.alpha = alpha

    def _initialize(self):
        self.xi = {}
        self.xi_t = {}
        self.m_ia = {}
        self.phi_gms = {}
        self.psi_gms = {}
        for psi in self.psis.factors:
            self.xi[psi] = {}
            for phi in self.phis.factors:
                # Now we only take those phis whose domain is a subset of the domain of psi
                # We could alternatively create a factor with the intersection of the domains whenever the factors overlap.
                # Indeed, I think that will be better.
                if phi.vars.issubset(psi.vars):
                    if phi not in self.xi_t:
                        self.xi_t[phi] = {}
                    self.xi[psi][phi] = _factor_maker(phi.vars)
                    self.xi_t[phi][psi] = self.xi[psi][phi]

        for phi in self.phis.factors:
            self.phi_gms[phi] = GraphModel(self.xi_t[phi].values(), copy=False, isLog=True)
        for psi in self.psis.factors:
            self.psi_gms[psi] = GraphModel(self.xi[psi].values(), copy=False, isLog=True)

        for phi in self.phis.factors:
            self.m_ia[phi] = {}
            m_ia = self.phi_gms[phi].joint()
            for psi in self.xi_t[phi]:
                self.m_ia[phi][psi] = m_ia - self.xi[psi][phi]

        self._compute_phis_from_xi()

    def _compute_phis_from_xi(self):
        for phi in self.phis.factors:
            phi.table = self.phi_gms[phi].joint().table

    def _update(self):
        for psi in self.psis.factors:
            self._project_psi(psi)
        #self._compute_phis_from_xi()
        return self.phis

    def _update_m_ia(self, i, a):
        #phi = i    ;    psi = a
        self.m_ia[i][a] = self.phi_gms[i].joint()
        self.m_ia[i][a] -= self.xi[a][i]

    def _m_ai(self, a, i):
        #phi : i   ;   psi : a
        return self.xi[a][i]

    def _sp_a(self, psi):
        f = self.alpha*psi
        for phi in self.xi[psi]:
            f += (1-self.alpha)*self._m_ai(psi,phi) + self.m_ia[phi][psi]
        return f.sum()

    def _xi_marginal(self, psi, phi):
        f = self.alpha*psi
        for phi_p in self.xi[psi]:
            if phi != phi_p:
                f += (1-self.alpha)*self._m_ai(psi,phi_p) + self.m_ia[phi_p][psi]
        return f.lsemarginal(phi.vars)

    def _project_psi(self, psi):
        sp = self._sp_a(psi)

        m_ais = {}
        for phi in self.xi[psi]:
            self._update_m_ia(phi, psi)
        for phi in self.xi[psi]:
            m_ai = self._m_ai(psi, phi)
            marg = self._xi_marginal(psi, phi)
            num = (1-self.alpha) * m_ai + self.m_ia[phi][psi] + marg #num = proj[num]
            # TODO: which version? no se aprecian diferencias significativas
            den = self.m_ia[phi][psi] + sp
            m_ais[phi] = num-den
            #m_ais[phi] = num.normalizeLog() - self.m_ia[phi][psi]
            #m_ais[phi].normalizeLogIP()

        for phi in self.xi[psi]:
            m_ai = self._m_ai(psi, phi)
            phi -= m_ai
            m_ai.table = self.step*m_ai.table + (1-self.step)*m_ais[phi].table
            m_ai.normalizeLogIP()
            phi += m_ai

    @classmethod
    def straight_project(cls, p:GraphModel, family=None, alpha=default_r, **kwargs):
                         #step = default_step, tol=default_tol, num_it=default_num_it,
                         #verbose=False, normalize_qs=True):
        # TODO: study effect of 'step'
        proj_method = cls(p, family, alpha)
        q_model = proj_method.project(**kwargs)#step, tol, num_it, verbose, normalize_qs)
        q_model.toExp()
        return q_model, proj_method.is_converged()


"""
Fully Factored (MF) I-projection
"""
class IProjection(Projection):
    def __init__(self, p, family=None):
        Projection.__init__(self, p, family)

    def _initialize(self):
        self.xi = {}
        self.xi_t = {}
        self.phi_gms = {}
        self.psi_gms = {}
        for psi in self.psis.factors:
            self.xi[psi] = []
            for phi in self.phis.factors:
                # Now we only take those phis whose domain is a subset of the domain of psi
                # We could alternatively create a factor with the intersection of the domains whenever the factors overlap.
                # Indeed, I think that will be better.
                if phi.vars.issubset(psi.vars):
                    if phi not in self.xi_t:
                        self.xi_t[phi] = []
                    self.xi[psi].append(phi)
                    self.xi_t[phi].append(psi)

        for phi in self.phis.factors:
            self.phi_gms[phi] = GraphModel(self.xi_t[phi], copy=False, isLog=True)
        for psi in self.psis.factors:
            self.psi_gms[psi] = GraphModel(self.xi[psi], copy=False, isLog=True)


    def _update(self):
        for phi in self.phis.factors:
            self._project_on_phi(phi)
        #self._compute_phis_from_xi()
        return self.phis

    def _phis_wo_phi_for(self, phi, psi):
        return self.psi_gms[psi].joint()-phi

    def _project_on_phi(self, phi):
        f_res = Factor(phi.v, 0.0)
        for psi in self.xi_t[phi]:
            #print("ai",psi.t)
            n_phi = self._phis_wo_phi_for(phi, psi)
            n_phi.normalizeLogIP() # TODO: Normalize or/and add 1/d
            n_phi.expIP()
            for val in np.arange(phi.dims()[0]):
                cond_psi = psi.condition({phi.v[0]: val})
                #cond_psi.expIP()            # TODO: segun Koller, esto no deberia ser así (a)
                f_res.t[val] += (n_phi * cond_psi).sum()
        #f_res.logIP()                       # TODO: segun Koller, esto no deberia ser así (b)
        phi.t = f_res.t
        phi.normalizeLogIP()

    @classmethod
    def straight_project(cls, p:GraphModel, family=None, **kwargs):
        proj_method = cls(p, family)
        q_model = proj_method.project(**kwargs)
        q_model.toExp()
        return q_model, proj_method.is_converged()


"""
Fully Factored (MF) R-projection for Unfactored P distributions
"""
class UnfactoredRProjection(Projection):
    def __init__(self, p, family=None, r=default_r):
        Projection.__init__(self, p, family)
        self.r = r

    def _initialize(self):
        self.p_joint = self.psis.joint()

    def _update(self):
        for phi in self.phis.factors:
            self._project_on_phi(phi)
        #self._compute_phis_from_xi()
        return self.phis

    def _phis_wo_phi(self, phi):
        return self.phis.joint()-phi

    def _project_on_phi(self, phi):
        #np.sqrt(np.sum(fp_h_equals_v * fp_h_equals_v / q_except_h))

        n_phi = self._phis_wo_phi(phi)
        f_res = Factor(phi.v, 0.0)
        for val in np.arange(phi.dims()[0]):
            f_to_marg = self.p_joint.condition({phi.v[0]: val})
            #f_to_marg.normalizeLogIP()
            f_to_marg = self.r*f_to_marg+(1-self.r)*n_phi
            f_to_marg.expIP()
            f_res.t[val] = f_to_marg.sum()
            #f_res.t[val] = logsumexp(f_to_marg.t)

        f_res.logIP()
        f_res *= 1./self.r
        # TODO: si añadimos el log de Renyi, aqui cambia algo?
        f_res.normalizeLogIP()

        phi.t = f_res.t

    @classmethod
    def straight_project(cls, p:GraphModel, family=None, r=default_r, **kwargs):
        proj_method = cls(p, family, r)
        q_model = proj_method.project(**kwargs)
        q_model.toExp()
        return q_model, proj_method.is_converged()


"""
Renyi-projection single phi's sum of projections
"""
class SumFactoredRProjection(Projection):
    ''' psis_strategy=
    - "single" : each single factor
    - "marg" : complete marginal of the scope of each factor
    - "part_prod_marg" : product of marginals (each factor to the scope of the marg)
    - "part_marg_prod" : marginal of product of factors to the scope of the marg
    '''
    def __init__(self, p, family, r=default_r, psis_strategy="single"):
        Projection.__init__(self, p, family)
        self.psis_strategy = psis_strategy
        self.r = r

    def _initialize(self):
        self.xi = {}
        self.xi_t = {}
        #orig_psis = self.psis.copy()
        #joint_p = orig_psis.joint().exp()
        for ind_psi, psi in enumerate(self.psis.factors):
            '''if self.psis_strategy == "marg":
                # psi<-d(psi_j)
                psi.t = joint_p.marginal(psi.vars).log().t
            elif self.psis_strategy == "part_prod_marg":
                # psi_j *= prod_k ( psi_k<-d(psi_j) )
                for alt_ind_psi, alt_psi in enumerate(orig_psis.factors):
                    if ind_psi != alt_ind_psi and \
                            not psi.vars.isdisjoint(alt_psi.vars):
                        psi += orig_psis.factors[ind_psi].lsemarginal(psi.vars)
            elif self.psis_strategy == "part_marg_prod":
                # psi_j *= prod_k ( psi_k<-d(psi_j) )
                ini_vars = psi.vars.copy()
                for alt_ind_psi, alt_psi in enumerate(orig_psis.factors):
                    if ind_psi != alt_ind_psi and \
                            not ini_vars.isdisjoint(alt_psi.vars):
                        psi += orig_psis.factors[ind_psi]
                psi.lsemarginal(ini_vars)'''
            self.xi[psi] = []
            for phi in self.phis.factors:
                if phi.vars.issubset(psi.vars):
                    if phi not in self.xi_t:
                        self.xi_t[phi] = []
                    self.xi_t[phi].append(psi)
                    self.xi[psi].append(phi)
        for phi in self.phis.factors:
            if phi not in self.xi_t:
                print("[Warning] Wasted phi", phi)
                phi.t[:]=0

    def _update(self):
        # TODO: changed to avoid wasted phi's. is this ok?
        for phi in self.xi_t:#self.phis.factors:
            self._project_on_phi(phi)
        #self._compute_phis_from_xi()
        return self.phis

    def _Aj_marginal(self, psi, phi):
        #f = self.phis.joint() - phi
        #f = f.lsemarginal(psi.vars)
        f = Factor(phi.v, vals=0)
        for phi_b in self.xi[psi]:
            if phi != phi_b:
                f += phi_b
        f *= 1-self.r
        f += self.r*psi
        return f.lsemarginal(phi.vars)

    def _Sj_marginal(self, psi, phi):
        f = self.phis.joint()
        f = f.lsemarginal(psi.vars)
        f *= 1-self.r
        f += self.r*psi
        f.expIP()
        return f.sum()

    def _project_on_phi(self, phi):
        A = Factor(vals=0.0)

        for psi in self.xi_t[phi]:
            a_j = self._Aj_marginal(psi, phi).expIP()
            A += a_j
            # TODO: R-divergence includes a log in
            #s_j = self._Sj_marginal(psi,phi)         # intro log in Renyi
            #A += a_j/s_j                             # intro log in Renyi
        A.logIP()
        A *= 1./self.r
        A.normalizeLogIP()
        phi.table = self.step * phi.table + (1 - self.step) * A.table

    @classmethod
    def straight_project(cls, p:GraphModel, family=None, r=default_r, psis_strategy="single", **kwargs):
                         #step = default_step, tol=default_tol, num_it=default_num_it,
                         #verbose=False, normalize_qs=True):
        proj_method = cls(p, family, r, psis_strategy)
        q_model = proj_method.project(**kwargs)#step, tol, num_it, verbose, normalize_qs)
        q_model.toExp()
        return q_model, proj_method.is_converged()


"""
Renyi-projection single phi's
"""
class SPFactoredRProjection(Projection):
    def __init__(self, p, family, r=default_r, psis_strategy="entropy", limit_nvars=default_limit_nvars):
        Projection.__init__(self, p, family)
        self.psis_strategy = psis_strategy
        self.r = r
        self.limit_nvars = limit_nvars

    def _initialize(self):
        if self.psis_strategy == "entropy":
            self._initialize_entropy()
        elif self.psis_strategy == "second_hop":
            self._initialize_second_hop()
        else:
            self._initialize_first_hop()

    def _initialize_entropy(self):
        self.phi_to_psi = {}
        self.phi_to_pseudopsi = {}
        self.phi_to_otherphi = {}
        for ind_phi, phi in enumerate(self.phis.factors):
            #print("for", phi)
            self.phi_to_psi[phi] = []
            self.phi_to_otherphi[phi] = []

            inc_vars = phi.v.copy()
            stop = False
            while not stop:
                candidates = self.psis.factorsWithAny(inc_vars)
                sel_candidate = None
                sel_entropy = np.inf
                stop = True
                for psi in candidates:
                    if psi not in self.phi_to_psi[phi] and \
                            len(psi.v|inc_vars) <= self.limit_nvars:
                        act_ent = self.entropy(psi)
                        if act_ent < sel_entropy:
                            sel_entropy = act_ent
                            sel_candidate = psi
                            stop = False
                if not stop:
                    #print("add ps",sel_candidate)
                    self.phi_to_psi[phi].append(sel_candidate)
                    inc_vars |= sel_candidate.v
            self.phi_to_pseudopsi[phi] = self.pseudo_psi(phi)
            for ind_phi_b, phi_b in enumerate(self.phis.factors):
                if ind_phi != ind_phi_b and \
                        phi_b.vars.issubset(self.phi_to_pseudopsi[phi].vars):
                    self.phi_to_otherphi[phi].append(phi_b)

    def entropy(self, f):
        return f.exp().entropy()

    def _initialize_first_hop(self):
        self.phi_to_psi = {}
        self.phi_to_pseudopsi = {}
        self.phi_to_otherphi = {}
        for ind_phi, phi in enumerate(self.phis.factors):
            self.phi_to_psi[phi] = []
            self.phi_to_otherphi[phi] = []
            for psi in self.psis.factors:
                if phi.vars.issubset(psi.vars):
                    self.phi_to_psi[phi].append(psi)
            self.phi_to_pseudopsi[phi] = self.pseudo_psi(phi)
            for ind_phi_b, phi_b in enumerate(self.phis.factors):
                if ind_phi != ind_phi_b and \
                        phi_b.vars.issubset(self.phi_to_pseudopsi[phi].vars):
                    self.phi_to_otherphi[phi].append(phi_b)

    def _initialize_second_hop(self):
        self.phi_to_psi = {}
        self.phi_to_pseudopsi = {}
        self.phi_to_otherphi = {}
        for ind_phi, phi in enumerate(self.phis.factors):
            act_vars = phi.v.copy()
            for psi in self.psis.factors:
                if phi.vars.issubset(psi.vars):
                    act_vars |= psi.v
            self.phi_to_psi[phi] = self.psis.factorsWithAny(act_vars)
            self.phi_to_pseudopsi[phi] = self.pseudo_psi(phi)
            self.phi_to_otherphi[phi] = []
            for ind_phi_b, phi_b in enumerate(self.phis.factors):
                if ind_phi != ind_phi_b and \
                        phi_b.vars.issubset(self.phi_to_pseudopsi[phi].vars):
                    self.phi_to_otherphi[phi].append(phi_b)

    def pseudo_psi(self, phi):
        f = Factor(vals=0.0)
        for psi in self.phi_to_psi[phi]:
            f += psi
        f *= self.r

        return f

    def _update(self):
        for phi in self.phi_to_pseudopsi:
            self._project_on_phi(phi)
        return self.phis

    def _project_on_phi(self, phi):
        f = self.phi_to_pseudopsi[phi].copy()

        for phi_b in self.phi_to_otherphi[phi]:
            f += (1-self.r)*phi_b

        f = (1./self.r)*f.lsemarginal(target=phi.v)
        f.normalizeLogIP()
        phi.table = self.step * phi.table + (1 - self.step) * f.table # f.table #

    @classmethod
    def straight_project(cls, p:GraphModel, family=None, r=default_r, psis_strategy="entropy",
                         limit_nvars=default_limit_nvars, **kwargs):#step=default_step, tol=default_tol,
                         #num_it=default_num_it, verbose=False, normalize_qs=True):
        proj_method = cls(p, family, r, psis_strategy, limit_nvars)
        q_model = proj_method.project(**kwargs)#step, tol, num_it, verbose, normalize_qs)
        q_model.toExp()
        return q_model, proj_method.is_converged()

#################################################################################################
###################################### Sampling functions #######################################
#################################################################################################

def sample_from_MF(model, num_samples):
    """
    Sample from a model which is known to be fully factorized
    :param model: model to sample from (MF)
    :param num_samples:  number of samples to obtain
    :return: list of samples, list of corresponding logProbs
    """
    samples = np.zeros((num_samples, model.nvar), dtype=int)
    samp_probs = np.zeros(num_samples)

    for f in model.factors:
        v = f.vars[0].label
        probs = f.t / f.sum()
        cprobs = np.cumsum(probs)

        samples[:, v] = np.argmax(np.random.random((num_samples,1)) < cprobs[np.newaxis, :], axis=1)
        samp_probs += np.log(probs[samples[:, v]])

    return samples, samp_probs

def probs_from_MF(model, samples):
    """
    Sample from a model which is known to be fully factorized
    :param model: model to sample from (MF)
    :param samples: samples to obtain probabilities from
    :return: list of logProbs
    """
    samp_probs = np.zeros(samples.shape[0])

    for f in model.factors:
        v = f.vars[0].label
        probs = f.t / f.sum()

        samp_probs += np.log(probs[samples[:, v]])

    return samp_probs


#################################################################################################
########################################## Algorithms ###########################################
#################################################################################################

class SamplingBasedEstimation:
    """
    General abstract class for sampling based estimation methods
    """
    def __init__(self, name):
        self.name = name

    def set_problem(self, f):
        self.f = f

    def set_real_value(self, val):
        self.real_value = val

    def estimate(self, num_samples, num_measures):
        raise Exception("Not implemented")

    def theoretical_variance(self, num_samples, num_measures):
        raise Exception("Not implemented")


class DeterministicEstimation(SamplingBasedEstimation):
    def __init__(self, ext_name="", random_proposal=False):
        SamplingBasedEstimation.__init__(self, "DEst")
        self.name = self.name + ext_name
        self.random_proposal = random_proposal

    def get_samples(self, num_samples=1000):
        cardinalities = np.array([x.states for x in self.f.X])
        space_size = np.prod(cardinalities)

        samples = np.zeros((space_size, self.f.nvar), dtype=int)
        r = space_size
        t = 1
        for i_x, card_x in enumerate(cardinalities):
            r /= card_x
            samples[:,i_x] = np.tile(np.repeat(np.arange(card_x,dtype=int),r),t)
            t *= card_x

        inds = np.arange(space_size)
        np.random.shuffle(inds)
        if num_samples < space_size:
            inds = inds[:num_samples]

        probs = np.zeros(space_size)-np.log(space_size)

        return samples[inds,:], probs[inds]

    def estimate(self, num_samples=1000, num_measures=100):
        if num_samples % num_measures != 0:
            raise Exception("The number of measures should be a multiple of the number of samples")

        samples, probs_q = self.get_samples(num_samples)

        probs_f = []
        for config in samples:
            prob = self.f.logValue(config)
            probs_f.append(prob)
        probs_f = np.array(probs_f)

        samples_pp = num_samples // num_measures
        samples_at_point = samples_pp * (np.arange(num_measures) + 1)
        if self.random_proposal:
            point_estimates = np.exp(probs_f - probs_q)
            estimates = [np.log(np.mean(point_estimates[0: p])) for p in samples_at_point]
        else:
            point_estimates = np.exp(probs_f)
            estimates = [np.log(np.sum(point_estimates[0: p])) for p in samples_at_point]
        variances = None
        if self.real_value is not None:
            variances = samples_at_point * ((estimates - self.real_value) ** 2.)

        return estimates, variances


class SimpleMonteCarlo(SamplingBasedEstimation):
    def __init__(self, ext_name="", random_proposal=False):
        SamplingBasedEstimation.__init__(self, "sMC")
        self.name = self.name + ext_name
        self.random_proposal = random_proposal

    def get_samples(self, num_samples=1000):
        samples = np.zeros((num_samples, self.f.nvar), dtype=int)

        cardinalities = np.array([x.states for x in self.f.X])
        for i_x, card_x in enumerate(cardinalities):
            samples[:,i_x] = np.random.randint(card_x, size=num_samples)

        probs = np.zeros(num_samples)-np.log(np.prod(cardinalities))

        return samples, probs

    def estimate(self, num_samples=1000, num_measures=100):
        if num_samples % num_measures != 0:
            raise Exception("The number of measures should be a multiple of the number of samples")

        samples, probs_q = self.get_samples(num_samples)

        probs_f = []
        for config in samples:
            prob = self.f.logValue(config)
            probs_f.append(prob)
        probs_f = np.array(probs_f)

        samples_pp = num_samples // num_measures
        samples_at_point = samples_pp * (np.arange(num_measures) + 1)
        if self.random_proposal:
            point_estimates = np.exp(probs_f - probs_q)
            estimates = [np.log(np.mean(point_estimates[0: p])) for p in samples_at_point]
        else:
            point_estimates = np.exp(probs_f)
            estimates = [np.log(np.sum(point_estimates[0: p])) for p in samples_at_point]
        variances = None
        if self.real_value is not None:
            variances = samples_at_point * ((estimates - self.real_value) ** 2.)

        return estimates, variances



class VIS(SamplingBasedEstimation):
    """
    General class for Variational Importance Sampling
    """
    def __init__(self, name, proj_function, ext_name="", **kwargs):
        SamplingBasedEstimation.__init__(self, name)
        self.proj_function = proj_function
        self.family_gen = lambda v: domains_k(p=v,k=1)
        self.sampling_function = sample_from_MF
        self.name = self.name + ext_name
        self.arguments = kwargs

    def set_problem(self, f):
        self.f = f
        if self.f is not None:
            self.project_onto_q()

    def project_onto_q(self):
        self.q, self.converged = self.proj_function(self.f, family=self.family_gen(self.f),verbose=False,
                                                    **self.arguments)
        print([f.table for f in self.q.factors])

    def get_convergence_distance(self):
        dist = self.f.toExp().joint().normalizeIP().distance(self.q.toExp().joint().normalizeIP(), "r2")
        return dist

    def is_projection_converged(self):
        return self.converged

    def get_samples(self, num_samples=1000, num_measures=100):
        samples, probs_q = self.sampling_function(self.q, num_samples)

        return samples, probs_q

    def estimate(self, num_samples=1000, num_measures=100):
        if num_samples % num_measures != 0:
            raise Exception("The number of measures should be a multiple of the number of samples")

        samples, probs_q = self.get_samples(num_samples, num_measures)

        probs_f = []
        for config in samples:
            prob = self.f.logValue(config)
            probs_f.append(prob)
        probs_f = np.array(probs_f)

        point_estimates = np.exp(probs_f - probs_q)


        samples_pp = num_samples // num_measures
        samples_at_point = samples_pp * (np.arange(num_measures) + 1)
        estimates = [np.log(np.mean(point_estimates[0: p])) for p in samples_at_point]
        variances = None
        if self.real_value is not None:
            variances = samples_at_point * ((estimates - self.real_value) ** 2.)

        return estimates, variances

#    def estimate(self, num_samples, num_measures, real_val=None, **kwargs):
#        return importance_sampling(self.f, self.q, num_samples, num_measures, real_val, **kwargs)

    def theoretical_variance(self, num_samples, num_measures):
        """
        def importance_sampling_theoretical_variance(proposal, function, importance_factor, real_val, num_samples=1000,
                                                     num_measures=100):
        samples_per_measure = num_samples // num_measures
        theo_var = (function * importance_factor - real_val) ** 2 * proposal
        theo_var = np.sum(theo_var[~np.isclose(proposal, 0., atol=1e-6/len(proposal))])
        theo_variances = theo_var / ((np.arange(num_measures) + 1) * samples_per_measure)

        return theo_variances
        """
        raise Exception("Not implemented")


class VIS_I(VIS):
    def __init__(self, **kwargs):
        VIS.__init__(self, "VIS-I", IProjection.straight_project, **kwargs)

class VIS_A(VIS):
    def __init__(self, **kwargs):
        VIS.__init__(self, "VIS-A", AlphaProjection.straight_project, **kwargs)

class VIS_Ru(VIS):
    def __init__(self, **kwargs):
        VIS.__init__(self, "VIS-R", UnfactoredRProjection.straight_project, **kwargs)

class VIS_R(VIS):
    def __init__(self, **kwargs):
        VIS.__init__(self, "VIS-Rm", FactoredRProjectionV2.straight_project, **kwargs)

class VIS_R_sp(VIS):
    def __init__(self, **kwargs):
        VIS.__init__(self, "VIS-Rh", SPFactoredRProjection.straight_project, **kwargs)

class VIS_R_sum(VIS):
    def __init__(self, **kwargs):
        VIS.__init__(self, "VIS-R-sum", SumFactoredRProjection.straight_project, **kwargs)


class VIS_Sp_I(VIS):
    def __init__(self, name="VIS-Rh-I", rate_i=0.01, fixed_i=False, control_var=False, deterministic_i=False, **kwargs):
        VIS.__init__(self, name, SPFactoredRProjection.straight_project, **kwargs)

        self.rate_i = rate_i
        self.fixed_i = fixed_i
        self.control_variates = control_var
        self.deterministic_i = deterministic_i

    def project_onto_q(self):
        self.q, self.converged = self.proj_function(self.f, family=self.family_gen(self.f),verbose=False,
                                                    **self.arguments)
        args_2 = self.arguments.copy()
        del args_2['r']
        self.q_i, self.converged_i = IProjection.straight_project(self.f, family=self.family_gen(self.f),verbose=False,
                                                    **args_2)

        if self.fixed_i:
            for f in self.q_i.factors:
                f.t = np.round(f.t,0)

        inst = np.zeros(self.q_i.nvar,dtype=int)
        l_probs = np.zeros(self.q_i.nvar)
        for f in self.q_i.factors:
            x = f.vars[0].label
            p_max = np.max(f.t)
            v_max = np.argmax(f.t)
            inst[x] = v_max
            l_probs[x] = p_max
        print("x =",inst)
        print("q_i(x_i) =",l_probs)
        print("q(x) =",np.prod(l_probs))
        print("p_U(x) =", 1/(2**self.q_i.nvar))
        print("p(x) =",np.exp(self.f.logValue(inst)-self.real_value))

        '''for f in self.q.factors:
            v = f.vars[0]
            f_i = self.q_i.factorsWith(v)[0]
            #print(f, f.t, f_i, f_i.t)
            f.t = (1.-self.rate_i)*f.t+self.rate_i*f_i.t
            #print(self.rate_i, f.t)'''
        #print([f.table for f in self.q.factors])

    def sample_and_evaluate(self, p, q, num_samples):
        samples, probs_p = self.sampling_function(p, num_samples)
        probs_q = probs_from_MF(q, samples)

        return samples, probs_p, probs_q

    def get_samples(self, num_samples=1000, num_measures=100, return_qis=False):
        num_samples_i = int(np.round(num_samples*self.rate_i))
        if not self.deterministic_i:
            num_samples_i = np.random.binomial(num_samples, self.rate_i, 1)[0]
        num_samples_r = num_samples-num_samples_i

        samples_r, probs_q_r_s_r, probs_q_i_s_r = self.sample_and_evaluate(self.q, self.q_i, num_samples_r)
        samples_i, probs_q_i_s_i, probs_q_r_s_i = self.sample_and_evaluate(self.q_i, self.q, num_samples_i)

        samples = np.vstack((samples_i, samples_r))
        probs_q_i = np.concatenate((probs_q_i_s_i, probs_q_i_s_r))
        probs_q_r = np.concatenate((probs_q_r_s_i, probs_q_r_s_r))
        probs_q = np.log(self.rate_i*np.exp(probs_q_i) + (1-self.rate_i)*np.exp(probs_q_r))

        if self.deterministic_i:
            cols_i = num_samples_i // num_measures
            cols_r = num_samples_r // num_measures
            aux_mat = np.ones((num_measures, num_samples // num_measures), dtype=int)*-1
            aux_mat[:, cols_i:(cols_i + cols_r)] = -2
            if cols_i + cols_r < aux_mat.shape[1]:
                last_i = np.ones(num_samples_i - cols_i * num_measures) * -1
                last_r = np.ones(num_samples_r - cols_r * num_measures) * -2
                last = np.concatenate((last_i, last_r))
                np.random.shuffle(last)
                aux_mat[:, -1] = last
            aux_vect = np.ravel(aux_mat)

            inds = np.argsort(np.concatenate((np.where(aux_vect == -1)[0], np.where(aux_vect == -2)[0])))

        else:
            inds = np.arange(num_samples)
            np.random.shuffle(inds)

        '''samples = samples[inds,:]
        probs_q_i = probs_q_i[inds]
        probs_q_r = probs_q_r[inds]
        probs_q = probs_q[inds]'''

        if return_qis:
            return samples[inds, :], probs_q[inds], probs_q_i[inds], probs_q_r[inds]
        else:
            return samples[inds, :], probs_q[inds]

    def estimate(self, num_samples=1000, num_measures=100):
        # Si NO usamos control variates, usamos el metodo clasico
        if not self.control_variates:
            return VIS.estimate(self,num_samples, num_measures)

        # si usamos control variates, hacemos....
        if num_samples % num_measures != 0:
            raise Exception("The number of measures should be a multiple of the number of samples")

        samples, probs_q, probs_q_i, probs_q_r = self.get_samples(num_samples, num_measures, return_qis=True)

        probs_f = []
        for config in samples:
            prob = self.f.logValue(config)
            probs_f.append(prob)
        probs_f = np.array(probs_f)

        samples_pp = num_samples // num_measures
        samples_at_point = samples_pp * (np.arange(num_measures) + 1)

        # calculate variates coefficients
        X = np.exp(np.column_stack((probs_q_i, probs_q_r)) - probs_q.reshape((len(probs_q), 1))) - 1
        y = np.exp(probs_f - probs_q)
        estimates = np.log([ LinearRegression().fit(X[:p, 1:], y[:p]).intercept_ for p in samples_at_point])
        #estimates = [np.log(np.mean(point_estimates[0: p])) for p in samples_at_point]
        variances = None
        if self.real_value is not None:
            variances = samples_at_point * ((estimates - self.real_value) ** 2.)

        return estimates, variances


class VIS_Sp_I_est(VIS_Sp_I):
    def __init__(self, **kwargs):
        VIS_Sp_I.__init__(self, name="VIS-Rh-I-est", **kwargs)
        self.samples_from = None

    def set_rate_i(self, rate_i=0.01):
        self.rate_i = rate_i

    def clean_sample_reserve(self):
        self.samples_from = None

    def sample_and_evaluate(self, p, q, num_samples):
        #print("a",num_samples)
        if self.samples_from is None:
            self.samples_from = {}
        if p not in self.samples_from:
            self.samples_from[p] = { "samples": np.zeros((0, p.nvar), dtype=int),
                                     "probs_p": np.array([]),
                                     "probs_q": np.array([]) }

        if self.samples_from[p]["samples"].shape[0] < num_samples:
            num_to_samp = num_samples - self.samples_from[p]["samples"].shape[0]
            #print("b",num_to_samp)
            samples, probs_p = self.sampling_function(p, num_to_samp)
            probs_q = probs_from_MF(q, samples)

            self.samples_from[p]["samples"] = np.vstack((self.samples_from[p]["samples"], samples))
            self.samples_from[p]["probs_p"] = np.concatenate((self.samples_from[p]["probs_p"], probs_p))
            self.samples_from[p]["probs_q"] = np.concatenate((self.samples_from[p]["probs_q"], probs_q))

        return self.samples_from[p]["samples"][:num_samples,:], self.samples_from[p]["probs_p"][:num_samples], \
               self.samples_from[p]["probs_q"][:num_samples]


class VIS_Sp_I_cont(VIS_Sp_I):
    def __init__(self, minoriting_factor=0.5, **kwargs):
        VIS_Sp_I.__init__(self, name="VIS-Rh-I-cont", deterministic_i=True, **kwargs)
        self.minoriting_factor = minoriting_factor

    def get_samples(self, num_samples=1000, num_measures=100):
        ### the self.deterministic_i is required in this approach. No way without deterministic sampling.
        samples_pp = num_samples // num_measures

        l_rates_i = np.array([self.rate_i*(self.minoriting_factor**e) for e in np.arange(num_measures)])
        #print("real",l_rates_i)
        l_num_samples_i = np.round(samples_pp*l_rates_i).astype(int)
        tot_num_samples_i = np.sum(l_num_samples_i)
        #num_samples_i = int(np.round(num_samples*self.rate_i))

        l_num_samples_r = samples_pp-l_num_samples_i
        tot_num_samples_r = np.sum(l_num_samples_r)

        samples_r, probs_q_r_s_r, probs_q_i_s_r = self.sample_and_evaluate(self.q, self.q_i, tot_num_samples_r)
        samples_i, probs_q_i_s_i, probs_q_r_s_i = self.sample_and_evaluate(self.q_i, self.q, tot_num_samples_i)

        samples = np.vstack((samples_i, samples_r))
        probs_q_i = np.concatenate((probs_q_i_s_i, probs_q_i_s_r))
        probs_q_r = np.concatenate((probs_q_r_s_i, probs_q_r_s_r))
        #probs_q = np.log(self.rate_i*np.exp(probs_q_i) + (1-self.rate_i)*np.exp(probs_q_r))

        aux_vect = np.ones(num_samples)*-2
        i_x = 0
        while l_num_samples_i[i_x] > 0 and i_x < num_measures:
            aux_vect[(i_x*samples_pp):(i_x*samples_pp+l_num_samples_i[i_x])] = -1
            i_x += 1
        inds = np.argsort(np.concatenate((np.where(aux_vect == -1)[0], np.where(aux_vect == -2)[0])))

        l_observed_rates_i = np.cumsum(l_num_samples_i)/np.array([samples_pp*(i+1) for i in np.arange(num_measures)])

        return samples[inds, :], probs_q_i[inds], probs_q_r[inds], l_observed_rates_i


    def estimate(self, num_samples=1000, num_measures=100):
        if num_samples % num_measures != 0:
            raise Exception("The number of measures should be a multiple of the number of samples")

        samples, probs_q_i, probs_q_r, l_observed_rates_i = self.get_samples(num_samples, num_measures)
        #print("obs",l_observed_rates_i)
        probs_f = []
        for config in samples:
            prob = self.f.logValue(config)
            probs_f.append(prob)
        probs_f = np.array(probs_f)

        samples_pp = num_samples // num_measures
        samples_at_point = samples_pp * (np.arange(num_measures) + 1)

        estimates = np.zeros(len(samples_at_point))

        # If we DO NOT use control variates...
        if not self.control_variates:
            for i_p, p in enumerate(samples_at_point):
                act_rate_i = l_observed_rates_i[i_p] # = np.mean(self.l_rates_i[:(i_p+1)])
                probs_q = np.log(act_rate_i * np.exp(probs_q_i[0: p]) + (1 - act_rate_i) * np.exp(probs_q_r[0: p]))

                estimates[i_p] = logsumexp(probs_f[0: p] - probs_q)

        else: # If we use control variates....

            # calculate variates coefficients
            for i_p, p in enumerate(samples_at_point):
                act_rate_i = l_observed_rates_i[i_p] # = np.mean(self.l_rates_i[:(i_p+1)])
                probs_q = np.log(act_rate_i * np.exp(probs_q_i) + (1 - act_rate_i) * np.exp(probs_q_r))
                X = np.exp(np.column_stack((probs_q_i, probs_q_r)) - probs_q.reshape((len(probs_q), 1))) - 1
                y = np.exp(probs_f - probs_q)
                estimates[i_p] = LinearRegression().fit(X[:p, 1:], y[:p]).intercept_
                #estimates = [np.log(np.mean(point_estimates[0: p])) for p in samples_at_point]
            estimates = np.log(estimates)

        variances = None
        if self.real_value is not None:
            variances = samples_at_point * ((estimates - self.real_value) ** 2.)

        return estimates, variances
