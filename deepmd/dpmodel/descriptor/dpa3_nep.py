# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
DPA3-NEP descriptor: replaces DPA3's RepFlow with NEP's descriptor.

NEP descriptor uses Chebyshev radial basis + spherical harmonic angular invariants
instead of the RepFlow iterative message passing architecture.

Reference:
  Zheyong Fan et al., Neuroevolution machine learning potentials ...
  Phys. Rev. B 104, 104309 (2021).
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
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.dpmodel.utils.update_sel import (
    UpdateSel,
)
from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.dpmodel.descriptor.descriptor import (
    extend_descrpt_stat,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.finetune import (
    get_index_between_two_maps,
    map_pair_exclude_types,
)
from deepmd.utils.version import (
    check_version_compatibility,
)
from deepmd.dpmodel.utils.network import (
    NativeLayer,
    get_activation_fn,
)
from deepmd.dpmodel.descriptor.nep_descriptor import (
    DescrptNEP,
    _cutoff_fn,
    _chebyshev_basis,
    _accumulate_s,
    _find_q_one,
    NUM_OF_ABC,
    C3B,
)


class NEPArgs:
    """Configuration for the NEP descriptor parameters."""

    def __init__(
        self,
        n_max_radial: int = 9,
        n_max_angular: int = 7,
        l_max: int = 4,
        basis_size_radial: int = 8,
        basis_size_angular: int = 8,
        rc_radial: float = 6.0,
        rc_angular: float = 4.0,
        rc_angular_smth: float = 3.5,
        sel_radial: int = 120,
        sel_angular: int = 20,
    ) -> None:
        self.n_max_radial = n_max_radial
        self.n_max_angular = n_max_angular
        self.l_max = l_max
        self.basis_size_radial = basis_size_radial
        self.basis_size_angular = basis_size_angular
        self.rc_radial = rc_radial
        self.rc_angular = rc_angular
        self.rc_angular_smth = rc_angular_smth
        self.sel_radial = sel_radial
        self.sel_angular = sel_angular

    def serialize(self) -> dict:
        return {
            "n_max_radial": self.n_max_radial,
            "n_max_angular": self.n_max_angular,
            "l_max": self.l_max,
            "basis_size_radial": self.basis_size_radial,
            "basis_size_angular": self.basis_size_angular,
            "rc_radial": self.rc_radial,
            "rc_angular": self.rc_angular,
            "rc_angular_smth": self.rc_angular_smth,
            "sel_radial": self.sel_radial,
            "sel_angular": self.sel_angular,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "NEPArgs":
        return cls(**data)


class DescrptDPA3_NEPCompatible(NativeOP, BaseDescriptor):
    """DPA3-NEP descriptor: NEP descriptor inside DPA3's interface.

    Uses NEP's Chebyshev radial basis + spherical harmonic angular invariants,
    exposed through DPA3's 5-tuple return interface for compatibility.
    """

    def __init__(
        self,
        ntypes: int,
        # args for NEP descriptor
        nep: NEPArgs | dict,
        # kwargs for descriptor
        concat_output_tebd: bool = False,
        activation_function: str = "silu",
        precision: str = "float64",
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        trainable: bool = True,
        seed: int | list[int] | None = None,
        use_econf_tebd: bool = False,
        use_tebd_bias: bool = False,
        use_loc_mapping: bool = True,
        type_map: list[str] | None = None,
        add_chg_spin_ebd: bool = False,
        default_chg_spin: list[float] | None = None,
    ) -> None:
        super().__init__()

        def init_subclass_params(sub_data, sub_class):
            if isinstance(sub_data, dict):
                return sub_class(**sub_data)
            elif isinstance(sub_data, sub_class):
                return sub_data
            else:
                raise ValueError(
                    f"Input args must be a {sub_class.__name__} class or a dict!"
                )

        self.nep_args = init_subclass_params(nep, NEPArgs)
        self.activation_function = activation_function
        self.use_econf_tebd = use_econf_tebd
        self.add_chg_spin_ebd = add_chg_spin_ebd
        if default_chg_spin is not None and len(default_chg_spin) != 2:
            raise ValueError("default_chg_spin must have exactly 2 values [charge, spin]")
        self.default_chg_spin = default_chg_spin
        self.use_tebd_bias = use_tebd_bias
        self.use_loc_mapping = use_loc_mapping
        self.type_map = type_map
        self.exclude_types = exclude_types
        self.env_protection = env_protection
        self.trainable = trainable

        # Create the NEP descriptor
        self.nep_descriptor = DescrptNEP(
            n_max_radial=self.nep_args.n_max_radial,
            n_max_angular=self.nep_args.n_max_angular,
            l_max=self.nep_args.l_max,
            basis_size_radial=self.nep_args.basis_size_radial,
            basis_size_angular=self.nep_args.basis_size_angular,
            rc_radial=self.nep_args.rc_radial,
            rc_angular=self.nep_args.rc_angular,
            rcut_smth=self.nep_args.rc_angular_smth,
            sel=self.nep_args.sel_radial,
            ntypes=ntypes,
            seed=child_seed(seed, 1),
            use_type_embedding=False,
            tebd_dim=0,
        )
        self.ntypes = ntypes
        self.rcut = self.nep_descriptor.get_rcut()
        self.rcut_smth = self.nep_descriptor.get_rcut_smth()
        self.sel = self.nep_descriptor.get_sel()

        # Type embedding
        self.tebd_dim = self.nep_args.n_max_radial + 1  # match radial descriptor dim
        self.type_embedding = TypeEmbedNet(
            ntypes=ntypes,
            neuron=[self.tebd_dim],
            padding=True,
            activation_function="Linear",
            precision=precision,
            use_econf_tebd=self.use_econf_tebd,
            use_tebd_bias=use_tebd_bias,
            type_map=type_map,
            seed=child_seed(seed, 2),
            trainable=trainable,
        )
        self.concat_output_tebd = concat_output_tebd
        self.precision = precision

        if self.add_chg_spin_ebd:
            self.cs_activation_fn = get_activation_fn(activation_function)
            self.chg_embedding = TypeEmbedNet(
                ntypes=200, neuron=[self.tebd_dim], padding=True,
                activation_function="Linear", precision=precision,
                seed=child_seed(seed, 3),
            )
            self.spin_embedding = TypeEmbedNet(
                ntypes=100, neuron=[self.tebd_dim], padding=True,
                activation_function="Linear", precision=precision,
                seed=child_seed(seed, 4),
            )
            self.mix_cs_mlp = NativeLayer(
                2 * self.tebd_dim, self.tebd_dim,
                precision=precision, seed=child_seed(seed, 5),
            )
        else:
            self.chg_embedding = self.spin_embedding = self.mix_cs_mlp = None

    def get_rcut(self) -> float:
        return self.rcut

    def get_rcut_smth(self) -> float:
        return self.rcut_smth

    def get_sel(self) -> list[int]:
        return self.sel

    def get_ntypes(self) -> int:
        return self.ntypes

    def get_type_map(self) -> list[str]:
        return self.type_map or []

    def get_dim_out(self) -> int:
        ret = self.nep_descriptor.get_dim_out()
        if self.concat_output_tebd:
            ret += self.tebd_dim
        return ret

    def get_dim_emb(self) -> int:
        return 0

    def mixed_types(self) -> bool:
        return True

    def has_message_passing(self) -> bool:
        return False

    def has_message_passing_across_ranks(self) -> bool:
        return False

    def need_sorted_nlist_for_lower(self) -> bool:
        return False

    def get_env_protection(self) -> float:
        return 0.0

    def get_dim_chg_spin(self) -> int:
        return 2 if self.add_chg_spin_ebd else 0

    def has_default_chg_spin(self) -> bool:
        return self.default_chg_spin is not None

    def get_default_chg_spin(self):
        return self.default_chg_spin

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
        """Compute NEP descriptor with DPA3-compatible output."""
        xp = array_api_compat.array_namespace(coord_ext, atype_ext, nlist)
        nframes, nloc, nnei = nlist.shape

        # Use the NEP descriptor
        node_ebd, rot_mat, edge_ebd, h2, sw = self.nep_descriptor.call(
            coord_ext, atype_ext, nlist, mapping=mapping,
            comm_dict=comm_dict, charge_spin=charge_spin,
        )

        if self.concat_output_tebd:
            # Get type embedding for concatenation
            type_emb = self.type_embedding.call()  # ntypes x tebd_dim
            atype_np = to_numpy_array(atype_ext).astype(int)
            tebd_np = np.zeros((nframes, nloc, self.tebd_dim), dtype=np.float64)
            for f in range(nframes):
                for i in range(nloc):
                    t = atype_np[f, i]
                    if t < type_emb.shape[0]:
                        tebd_np[f, i] = to_numpy_array(type_emb[t])
            tebd_arr = xp.asarray(tebd_np, dtype=coord_ext.dtype)
            node_ebd = xp.concat([node_ebd, tebd_arr], axis=-1)

        return node_ebd, rot_mat, edge_ebd, h2, sw

    def serialize(self) -> dict:
        nep = self.nep_descriptor
        data = {
            "@class": "Descriptor",
            "type": "dpa3_nep",
            "@version": 2,
            "ntypes": self.ntypes,
            "nep_args": self.nep_args.serialize(),
            "concat_output_tebd": self.concat_output_tebd,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "exclude_types": self.exclude_types,
            "env_protection": self.env_protection,
            "trainable": self.trainable,
            "seed": self.seed if hasattr(self, 'seed') else None,
            "use_econf_tebd": self.use_econf_tebd,
            "use_tebd_bias": self.use_tebd_bias,
            "use_loc_mapping": self.use_loc_mapping,
            "type_map": self.type_map,
            "type_embedding": self.type_embedding.serialize() if hasattr(self, 'type_embedding') else None,
            "nep_variable": {
                "c_radial": to_numpy_array(nep.c_radial).tolist(),
                "c_angular": to_numpy_array(nep.c_angular).tolist(),
                "q_scaler": to_numpy_array(nep.q_scaler).tolist(),
                "@variables": {
                    "davg": to_numpy_array(nep.mean).tolist(),
                    "dstd": to_numpy_array(nep.stddev).tolist(),
                },
            },
        }
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA3_NEPCompatible":
        data = data.copy()
        version = data.pop("@version", 1)
        check_version_compatibility(version, 2, 1)
        nep_variable = data.pop("nep_variable", {})
        type_embedding_data = data.pop("type_embedding", None)
        chg_embedding_data = data.pop("chg_embedding", None)
        spin_embedding_data = data.pop("spin_embedding", None)

        data["nep"] = NEPArgs(**data.pop("nep_args"))
        obj = cls(**data)

        if type_embedding_data:
            obj.type_embedding = TypeEmbedNet.deserialize(type_embedding_data)

        # Restore NEP variables
        if "c_radial" in nep_variable:
            obj.nep_descriptor.c_radial = np.array(nep_variable["c_radial"])
        if "c_angular" in nep_variable:
            obj.nep_descriptor.c_angular = np.array(nep_variable["c_angular"])
        if "q_scaler" in nep_variable:
            obj.nep_descriptor.q_scaler = np.array(nep_variable["q_scaler"])

        stat = nep_variable.get("@variables", {})
        if "davg" in stat:
            obj.nep_descriptor.mean = np.array(stat["davg"])
        if "dstd" in stat:
            obj.nep_descriptor.stddev = np.array(stat["dstd"])

        return obj

    def compute_input_stats(self, merged, path=None):
        pass

    def set_stat_mean_and_stddev(self, mean, stddev):
        self.nep_descriptor.mean = np.array(mean[0], dtype=np.float64)
        self.nep_descriptor.stddev = np.array(stddev[0], dtype=np.float64)

    def get_stat_mean_and_stddev(self):
        return [self.nep_descriptor.mean], [self.nep_descriptor.stddev]

    @classmethod
    def update_sel(cls, global_jdata, local_jdata):
        local_jdata_cpy = local_jdata.copy()
        update_sel = UpdateSel()
        nep = local_jdata_cpy.get("nep", {})
        min_nbor_dist, sel_new = update_sel.update_one_sel(
            nep.get("rc_radial", 6.0),
            nep.get("sel_radial", 120),
        )
        nep["sel_radial"] = sel_new[0]
        local_jdata_cpy["nep"] = nep
        return local_jdata_cpy, min_nbor_dist

    def dim_out(self) -> int:
        return self.get_dim_out()

    def dim_emb(self) -> int:
        return self.get_dim_emb()
