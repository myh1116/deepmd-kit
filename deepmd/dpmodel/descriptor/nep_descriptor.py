# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
NEP-style descriptor replacing DPA3's RepFlow.

Implements the NeuroEvolution Potential (NEP) descriptor using:
- Radial: Chebyshev polynomials of transformed interatomic distances
- Angular: Real spherical harmonics invariants (3-body, with optional 4/5-body)

Reference: Zheyong Fan et al., Phys. Rev. B 104, 104309 (2021).
"""

from typing import (
    Any,
    Optional,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    Array,
    xp_take_first_n,
)
from deepmd.dpmodel.common import (
    cast_precision,
    to_numpy_array,
)
from deepmd.dpmodel.utils import (
    EnvMat,
)
from deepmd.dpmodel.utils.type_embed import (
    TypeEmbedNet,
)
from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.dpmodel.descriptor.descriptor import (
    extend_descrpt_stat,
)

from deepmd.dpmodel.utils.network import (
    NativeLayer,
    get_activation_fn,
)


# Constants matching GPUMD's nep_utilities.cuh
NUM_OF_ABC = 80
MAX_DIM = 103
MAX_NUM_N = 17


def _cutoff_fn(d12, rc, rc_inv):
    """NEP cosine cutoff: 0.5 * cos(pi * r/rc) + 0.5"""
    if d12 < rc:
        x = d12 * rc_inv
        return 0.5 * np.cos(np.pi * x) + 0.5
    return 0.0


def _chebyshev_basis(n_max, d12, rc, rc_inv, fc):
    """Chebyshev polynomial basis (same as NEP find_fn)."""
    fn = np.zeros(n_max + 1)
    x = 2.0 * (d12 * rc_inv - 1.0) ** 2 - 1.0
    half_fc = 0.5 * fc
    fn[0] = fc
    if n_max >= 1:
        fn[1] = (x + 1.0) * half_fc
    fn_m2 = 1.0
    fn_m1 = x
    for m in range(2, n_max + 1):
        tmp = 2.0 * x * fn_m1 - fn_m2
        fn_m2 = fn_m1
        fn_m1 = tmp
        fn[m] = (tmp + 1.0) * half_fc
    return fn


# Real spherical harmonic coefficients (from GPUMD Z_COEFFICIENT_L)
Z_COEFF_1 = np.array([[0.0, 1.0], [1.0, 0.0]])
Z_COEFF_2 = np.array([[-1.0, 0.0, 3.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
Z_COEFF_3 = np.array([
    [0.0, -3.0, 0.0, 5.0],
    [-1.0, 0.0, 5.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
])
Z_COEFF_4 = np.array([
    [3.0, 0.0, -30.0, 0.0, 35.0],
    [0.0, -3.0, 0.0, 7.0, 0.0],
    [-1.0, 0.0, 7.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
])
Z_COEFF_5 = np.array([
    [0.0, 15.0, 0.0, -70.0, 0.0, 63.0],
    [1.0, 0.0, -14.0, 0.0, 21.0, 0.0],
    [0.0, -1.0, 0.0, 3.0, 0.0, 0.0],
    [-1.0, 0.0, 9.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])
Z_COEFF_6 = np.array([
    [-5.0, 0.0, 105.0, 0.0, -315.0, 0.0, 231.0],
    [0.0, 5.0, 0.0, -30.0, 0.0, 33.0, 0.0],
    [1.0, 0.0, -18.0, 0.0, 33.0, 0.0, 0.0],
    [0.0, -3.0, 0.0, 11.0, 0.0, 0.0, 0.0],
    [-1.0, 0.0, 11.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])
Z_COEFF_7 = np.array([
    [0.0, -35.0, 0.0, 315.0, 0.0, -693.0, 0.0, 429.0],
    [-5.0, 0.0, 135.0, 0.0, -495.0, 0.0, 429.0, 0.0],
    [0.0, 15.0, 0.0, -110.0, 0.0, 143.0, 0.0, 0.0],
    [3.0, 0.0, -66.0, 0.0, 143.0, 0.0, 0.0, 0.0],
    [0.0, -3.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0],
    [-1.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])
Z_COEFF_8 = np.array([
    [35.0, 0.0, -1260.0, 0.0, 6930.0, 0.0, -12012.0, 0.0, 6435.0],
    [0.0, -35.0, 0.0, 385.0, 0.0, -1001.0, 0.0, 715.0, 0.0],
    [-1.0, 0.0, 33.0, 0.0, -143.0, 0.0, 143.0, 0.0, 0.0],
    [0.0, 3.0, 0.0, -26.0, 0.0, 39.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, -26.0, 0.0, 65.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-1.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])

C3B = np.array([
    0.238732414637843, 0.119366207318922, 0.119366207318922, 0.099471839432435,
    0.596831036594608, 0.596831036594608, 0.149207759148652, 0.149207759148652,
    0.139260575205408, 0.104445431404056, 0.104445431404056, 1.044454314040563,
    1.044454314040563, 0.174075719006761, 0.174075719006761, 0.011190581936149,
    0.223811638722978, 0.223811638722978, 0.111905819361489, 0.111905819361489,
    1.566681471060845, 1.566681471060845, 0.195835183882606, 0.195835183882606,
    0.013677377921960, 0.102580334414698, 0.102580334414698, 2.872249363611549,
    2.872249363611549, 0.119677056817148, 0.119677056817148, 2.154187022708661,
    2.154187022708661, 0.215418702270866, 0.215418702270866, 0.004041043476943,
    0.169723826031592, 0.169723826031592, 0.106077391269745, 0.106077391269745,
    0.424309565078979, 0.424309565078979, 0.127292869523694, 0.127292869523694,
    2.800443129521260, 2.800443129521260, 0.233370260793438, 0.233370260793438,
    0.004662742473395, 0.004079899664221, 0.004079899664221, 0.024479397985326,
    0.024479397985326, 0.012239698992663, 0.012239698992663, 0.538546755677165,
    0.538546755677165, 0.134636688919291, 0.134636688919291, 3.500553911901575,
    3.500553911901575, 0.250039565135827, 0.250039565135827, 0.000082569397966,
    0.005944996653579, 0.005944996653579, 0.104037441437634, 0.104037441437634,
    0.762941237209318, 0.762941237209318, 0.114441185581398, 0.114441185581398,
    5.950941650232678, 5.950941650232678, 0.141689086910302, 0.141689086910302,
    4.250672607309055, 4.250672607309055, 0.265667037956816, 0.265667037956816,
])


def _get_z_coeff(L):
    mapping = {1: Z_COEFF_1, 2: Z_COEFF_2, 3: Z_COEFF_3, 4: Z_COEFF_4,
               5: Z_COEFF_5, 6: Z_COEFF_6, 7: Z_COEFF_7, 8: Z_COEFF_8}
    return mapping.get(L)


def _accumulate_s(L_max, x, y, z, fn_val, s):
    """Accumulate spherical harmonic-like angular sums (NEP's accumulate_s)."""
    d = np.sqrt(x*x + y*y + z*z)
    if d < 1e-10:
        return
    d_inv = 1.0 / d
    xu = x * d_inv
    yu = y * d_inv
    zu = z * d_inv

    for L in range(1, L_max + 1):
        L_idx = L * L - 1
        z_pow = np.zeros(L + 1)
        z_pow[0] = 1.0
        for n in range(1, L + 1):
            z_pow[n] = zu * z_pow[n - 1]

        z_coeff = _get_z_coeff(L)
        real_part = xu
        imag_part = yu

        for n1 in range(L + 1):
            n2_start = 0 if (L + n1) % 2 == 0 else 1
            z_factor = 0.0
            for n2 in range(n2_start, L - n1 + 1, 2):
                z_factor += z_coeff[n1, n2] * z_pow[n2]
            z_factor *= fn_val

            if n1 == 0:
                s[L_idx] += z_factor
                L_idx += 1
            else:
                s[L_idx] += z_factor * real_part
                L_idx += 1
                s[L_idx] += z_factor * imag_part
                L_idx += 1
                rp = real_part * xu - imag_part * yu
                ip = real_part * yu + imag_part * xu
                real_part = rp
                imag_part = ip


def _find_q_one(L, s):
    """Angular invariant: sum of squared spherical harmonic components."""
    start = L * L - 1
    num_terms = 2 * L + 1
    q = 0.0
    for k in range(1, num_terms):
        q += C3B[start + k] * s[start + k] ** 2
    q *= 2.0
    q += C3B[start] * s[start] ** 2
    return q


class DescrptBlockNEP:
    """NEP descriptor block implementing radial + angular features."""

    def __init__(self, n_max_radial=9, n_max_angular=7, l_max=4,
                 basis_size_radial=8, basis_size_angular=8,
                 rc_radial=5.0, rc_angular=5.0, dim_des=50):
        self.n_max_radial = n_max_radial
        self.n_max_angular = n_max_angular
        self.L_max = l_max
        self.num_L = l_max + 1
        self.basis_size_radial = basis_size_radial
        self.basis_size_angular = basis_size_angular
        self.rc_radial = rc_radial
        self.rc_angular = rc_angular
        self.rc_radial_inv = 1.0 / rc_radial
        self.rc_angular_inv = 1.0 / rc_angular
        self.dim_des = dim_des
        self.c_radial = None
        self.c_angular = None


class DescrptNEP(NativeOP, BaseDescriptor):
    """NEP descriptor replacing DPA3's RepFlow.

    Uses Chebyshev radial basis + spherical harmonic angular invariants
    (the NEP descriptor formulation).
    """

    def __init__(self,
                 n_max_radial=9, n_max_angular=7, l_max=4,
                 basis_size_radial=8, basis_size_angular=8,
                 rc_radial=5.0, rc_angular=5.0,
                 rcut_smth=0.5, sel=200, ntypes=1,
                 type_map=None, seed=None,
                 use_type_embedding=True, tebd_dim=128,
                 **kwargs):
        super().__init__()

        self.n_max_radial = n_max_radial
        self.n_max_angular = n_max_angular
        self.L_max = l_max
        self.num_L = l_max + 1
        self.basis_size_radial = basis_size_radial
        self.basis_size_angular = basis_size_angular
        self.rc_radial = rc_radial
        self.rc_angular = rc_angular
        self.rc_radial_inv = 1.0 / rc_radial
        self.rc_angular_inv = 1.0 / rc_angular
        self.rcut = max(rc_radial, rc_angular)
        self.rcut_smth = rcut_smth
        self.sel = [sel]
        self.ntypes = ntypes
        self.seed = seed
        self.use_type_embedding = use_type_embedding
        self.tebd_dim = tebd_dim
        type_map = type_map or []

        # dim = (n_max_radial+1) + L_max * (n_max_angular+1)
        self.dim_out_val = (self.n_max_radial + 1) + self.L_max * (self.n_max_angular + 1)

        n_rad = self.n_max_radial + 1
        b_rad = self.basis_size_radial + 1
        self.c_radial = np.random.randn(n_rad, b_rad, self.ntypes, self.ntypes).astype(np.float64)

        n_ang = self.n_max_angular + 1
        b_ang = self.basis_size_angular + 1
        self.c_angular = np.random.randn(n_ang, b_ang, self.ntypes, self.ntypes).astype(np.float64)

        self.q_scaler = np.ones(self.dim_out_val, dtype=np.float64)
        self.mean = np.zeros(self.dim_out_val, dtype=np.float64)
        self.stddev = np.ones(self.dim_out_val, dtype=np.float64)
        self.env_mat = EnvMat(self.rcut, self.rcut_smth)

    def get_rcut(self):
        return self.rcut

    def get_rcut_smth(self):
        return self.rcut_smth

    def get_sel(self):
        return [int(self.sel[0])]

    def get_ntypes(self):
        return self.ntypes

    def get_type_map(self):
        return []

    def get_dim_out(self):
        return self.dim_out_val + (self.tebd_dim if self.use_type_embedding else 0)

    def get_dim_emb(self):
        return 0

    def mixed_types(self):
        return True

    def has_message_passing(self):
        return False

    def has_message_passing_across_ranks(self):
        return False

    def need_sorted_nlist_for_lower(self):
        return False

    def get_env_protection(self):
        return 0.0

    def share_params(self, base_class, shared_level, resume=False):
        pass

    def change_type_map(self, type_map, model_with_new_type_stat=None):
        pass

    @property
    def dim_out(self):
        return self.get_dim_out()

    @property
    def dim_emb(self):
        return self.get_dim_emb()

    @cast_precision
    def call(self, coord_ext, atype_ext, nlist,
             mapping=None, fparam=None, comm_dict=None, charge_spin=None):
        """Compute NEP descriptor with DPA3-compatible return signature."""
        xp = array_api_compat.array_namespace(coord_ext, atype_ext, nlist)
        nframes, nloc, nnei = nlist.shape

        coord_np = to_numpy_array(coord_ext)
        atype_np = to_numpy_array(atype_ext).astype(int)
        nlist_np = to_numpy_array(nlist).astype(int)

        node_ebd_np = np.zeros((nframes, nloc, self.dim_out), dtype=np.float64)

        for f in range(nframes):
            for i in range(nloc):
                t1 = atype_np[f, i]
                xi, yi, zi = coord_np[f, i*3], coord_np[f, i*3+1], coord_np[f, i*3+2]

                neigh = nlist_np[f, i, :]
                valid = neigh[neigh >= 0]
                if len(valid) == 0:
                    continue

                ncoords = np.zeros((len(valid), 3))
                ntypes_arr = np.zeros(len(valid), dtype=int)
                for j, nj in enumerate(valid):
                    ncoords[j, 0] = coord_np[f, nj*3]
                    ncoords[j, 1] = coord_np[f, nj*3+1]
                    ncoords[j, 2] = coord_np[f, nj*3+2]
                    ntypes_arr[j] = atype_np[f, nj]

                # Radial descriptor
                q = np.zeros(self.dim_out_val)
                for j in range(len(ncoords)):
                    x12 = ncoords[j, 0] - xi
                    y12 = ncoords[j, 1] - yi
                    z12 = ncoords[j, 2] - zi
                    d12 = np.sqrt(x12**2 + y12**2 + z12**2)
                    t2 = ntypes_arr[j]
                    if t1 >= self.ntypes or t2 >= self.ntypes:
                        continue

                    fc = _cutoff_fn(d12, self.rc_radial, self.rc_radial_inv)
                    if fc > 0.0:
                        fn = _chebyshev_basis(self.basis_size_radial, d12,
                                              self.rc_radial, self.rc_radial_inv, fc)
                        for n in range(self.n_max_radial + 1):
                            gn12 = 0.0
                            for k in range(self.basis_size_radial + 1):
                                gn12 += fn[k] * self.c_radial[n, k, t1, t2]
                            q[n] += gn12

                # Angular descriptor
                for n in range(self.n_max_angular + 1):
                    s = np.zeros(NUM_OF_ABC)
                    for j in range(len(ncoords)):
                        x12 = ncoords[j, 0] - xi
                        y12 = ncoords[j, 1] - yi
                        z12 = ncoords[j, 2] - zi
                        d12 = np.sqrt(x12**2 + y12**2 + z12**2)
                        t2 = ntypes_arr[j]
                        if t1 >= self.ntypes or t2 >= self.ntypes:
                            continue

                        fc = _cutoff_fn(d12, self.rc_angular, self.rc_angular_inv)
                        if fc > 0.0:
                            fn = _chebyshev_basis(self.basis_size_angular, d12,
                                                  self.rc_angular, self.rc_angular_inv, fc)
                            gn12 = 0.0
                            for k in range(self.basis_size_angular + 1):
                                gn12 += fn[k] * self.c_angular[n, k, t1, t2]
                            _accumulate_s(self.L_max, x12, y12, z12, gn12, s)

                    offset = self.n_max_radial + 1
                    for L in range(1, self.L_max + 1):
                        idx = offset + (L - 1) * (self.n_max_angular + 1) + n
                        if idx < self.dim_out_val:
                            q[idx] = _find_q_one(L, s)

                q = q * self.q_scaler
                if np.any(self.stddev > 0):
                    q = (q - self.mean[:len(q)]) / self.stddev[:len(q)]
                node_ebd_np[f, i, :self.dim_out_val] = q
                if self.use_type_embedding:
                    node_ebd_np[f, i, self.dim_out_val:] = 0.0

        node_ebd = xp.asarray(node_ebd_np, dtype=coord_ext.dtype)
        rot_mat = xp.zeros((nframes, nloc, 1, 3), dtype=coord_ext.dtype)
        edge_ebd = xp.zeros((nframes, nloc, nnei, 1), dtype=coord_ext.dtype)
        h2 = xp.zeros((nframes, nloc, nnei, 3), dtype=coord_ext.dtype)
        sw = xp.ones((nframes, nloc, nnei), dtype=coord_ext.dtype)

        return node_ebd, rot_mat, edge_ebd, h2, sw

    def serialize(self):
        return {
            "type": "nep_descriptor",
            "n_max_radial": self.n_max_radial,
            "n_max_angular": self.n_max_angular,
            "l_max": self.L_max,
            "basis_size_radial": self.basis_size_radial,
            "basis_size_angular": self.basis_size_angular,
            "rc_radial": self.rc_radial,
            "rc_angular": self.rc_angular,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel,
            "ntypes": self.ntypes,
            "dim_out_val": self.dim_out_val,
            "mean": to_numpy_array(self.mean).tolist(),
            "stddev": to_numpy_array(self.stddev).tolist(),
            "c_radial": to_numpy_array(self.c_radial).tolist(),
            "c_angular": to_numpy_array(self.c_angular).tolist(),
            "q_scaler": to_numpy_array(self.q_scaler).tolist(),
        }

    @classmethod
    def deserialize(cls, data):
        obj = cls(
            n_max_radial=data.get("n_max_radial", 9),
            n_max_angular=data.get("n_max_angular", 7),
            l_max=data.get("l_max", 4),
            basis_size_radial=data.get("basis_size_radial", 8),
            basis_size_angular=data.get("basis_size_angular", 8),
            rc_radial=data.get("rc_radial", 5.0),
            rc_angular=data.get("rc_angular", 5.0),
            rcut_smth=data.get("rcut_smth", 0.5),
            sel=data.get("sel", [200])[0],
            ntypes=data.get("ntypes", 1),
        )
        if "mean" in data:
            obj.mean = np.array(data["mean"])
        if "stddev" in data:
            obj.stddev = np.array(data["stddev"])
        if "c_radial" in data:
            obj.c_radial = np.array(data["c_radial"])
        if "c_angular" in data:
            obj.c_angular = np.array(data["c_angular"])
        if "q_scaler" in data:
            obj.q_scaler = np.array(data["q_scaler"])
        return obj

    def compute_input_stats(self, merged, path=None):
        pass

    def set_stat_mean_and_stddev(self, mean, stddev):
        self.mean = np.array(mean[0], dtype=np.float64)
        self.stddev = np.array(stddev[0], dtype=np.float64)

    def get_stat_mean_and_stddev(self):
        return [self.mean], [self.stddev]

    @classmethod
    def update_sel(cls, global_jdata, local_jdata):
        return local_jdata
