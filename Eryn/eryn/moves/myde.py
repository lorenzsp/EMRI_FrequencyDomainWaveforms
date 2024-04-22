# -*- coding: utf-8 -*-

import numpy as np

from .red_blue import RedBlueMove

__all__ = ["MyDE"]


class MyDE(RedBlueMove):
    r"""A proposal using differential evolution.

    This `Differential evolution proposal
    <http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf>`_ is
    implemented following `Nelson et al. (2013)
    <https://arxiv.org/abs/1311.5229>`_.

    Args:
        sigma (float): The standard deviation of the Gaussian used to stretch
            the proposal vector.
        gamma0 (Optional[float]): The mean stretch factor for the proposal
            vector. By default, it is :math:`2.38 / \sqrt{2\,\mathrm{ndim}}`
            as recommended by the two references.

    """

    def __init__(self, sigma=1.0e-5, gamma0=None, **kwargs):
        self.sigma = sigma
        self.gamma0 = gamma0
        super(MyDE, self).__init__(**kwargs)


    def get_proposal(self, s_all, c_all, random, inds_s=None, inds_c=None):
        newpos = {}
        for i, name in enumerate(s_all):
            c = c_all[name]
            s = s_all[name]
            c = np.concatenate(c, axis=0)
            Ns, Nc = len(s), len(c)

            # gets rid of any values of exactly zero
            if inds_s is None:
                ndim_temp = s_all[name].shape[-1] * s_all[name].shape[-2]
            else:
                ndim_temp = inds_s[name].sum(axis=(1)) * s_all[name].shape[-1]
            if i == 0:
                ndim = ndim_temp
                Ns_check = Ns
                zz = random.rand(Ns)
            else:
                ndim += ndim_temp
                if Ns_check != Ns:
                    raise ValueError("Different number of walkers across models.")
            rint = random.randint(Nc, size=(Ns,))

            # check dimensioality comparison between s and c
            nleaves_s = inds_s[name].sum(axis=(1))
            nleaves_max = inds_s[name].shape[1]
            nleaves_c_rint = inds_c[name].sum(axis=(1))[rint]
            c_rint = c[rint]
            c_inds_rint = inds_c[name][rint]

            # same_num first
            temp = np.zeros_like(s)  # s.copy()
            for j in range(len(s)):
                # c_temp_inds = np.random.choice(
                #     np.arange(nleaves_max)[c_inds_rint[j].astype(bool)],
                #     int(nleaves_s[j]),
                #     replace=True,
                # )

                # temp[j, inds_s[name][j].astype(bool)] = c_rint[j][c_temp_inds]

                # my proposal
                arr = np.delete(np.arange(Ns),j)
                random.shuffle(arr)
                A,B,C = s[arr[:3]]
                R = random.randint(temp[j].shape[1])

                r_i = np.random.uniform(0,1,size=temp[j].shape)
                mask = (r_i<0.9)
                temp[j,mask] = A[mask] + 0.8*(B[mask] - C[mask])
                temp[j,~mask] = s[j,~mask]
                temp[j,:,R] = A[0,R] + 0.8*(B[0,R] - C[0,R])

            newpos[name] = temp

            if self.periodic is not None:
                newpos[name] = self.periodic.wrap(newpos[name], names=[name])[name]

        # proper factors
        factors = np.zeros_like(zz)
        return newpos, factors
