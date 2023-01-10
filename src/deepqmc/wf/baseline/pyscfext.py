import logging

import numpy as jnp
from pyscf import gto
from pyscf.mcscf import CASSCF
from pyscf.scf import RHF

log = logging.getLogger(__name__)


def pyscf_from_mol(mol, basis, cas=None, **kwargs):
    r"""Create a pyscf molecule and perform an SCF calculation on it.

    Args:
        mol (~deepqmc.Molecule): the molecule on which to perform the SCF calculation.
        basis (str): the name of the Gaussian basis set to use.
        cas (Tuple[int,int]): optional, the active space definition for CAS.

    Returns:
        tuple: the pyscf molecule and the SCF calculation object.
    """
    for atomic_number in mol.charges[jnp.invert(mol.pp_mask)].tolist():
        assert atomic_number not in mol.charges[mol.pp_mask], (
            'Usage of different pseudopotentials for atoms of the same element is not'
            ' implemented for pretraining.'
        )
    mol = gto.M(
        atom=mol.as_pyscf(),
        unit='bohr',
        basis=basis,
        charge=mol.charge,
        spin=mol.spin,
        cart=True,
        parse_arg=False,
        ecp=dict(
            zip(
                mol.charges[mol.pp_mask].astype(int).tolist(),
                [mol.pp_type] * int(mol.pp_mask.sum()),
            )
        ),
        verbose=0,
        **kwargs,
    )
    log.info('Running HF...')
    mf = RHF(mol)
    mf.kernel()
    log.info(f'HF energy: {mf.e_tot}')
    if cas:
        log.info('Running MCSCF...')
        mc = CASSCF(mf, *cas)
        mc.kernel()
        log.info(f'MCSCF energy: {mc.e_tot}')
    return mol, (mf, mc if cas else None)


def confs_from_mc(mc, tol=0):
    r"""Retrieve the electronic configurations contributing to a pyscf CAS-SCF solution.

    Args:
        mc: a pyscf MC-SCF object.
        tol (float): default 0, the CI weight threshold.

    Returns:
        list: the list of configurations in deepqmc format,
        with weight larger than :data:`tol`.
    """
    conf_coeff, *confs = zip(
        *mc.fcisolver.large_ci(mc.ci, mc.ncas, mc.nelecas, tol=tol, return_strs=False)
    )
    confs = [
        [
            jnp.tile(jnp.arange(mc.ncore), (len(conf_coeff), 1)),
            jnp.array(cfs) + mc.ncore,
        ]
        for cfs in confs
    ]
    confs = jnp.concatenate([jnp.concatenate(cfs, axis=-1) for cfs in confs], axis=-1)
    confs = sorted(zip(conf_coeff, confs), key=lambda x: -x[0] ** 2)
    return confs


def load_pp_param(charge, pp_type):
    """Load the pseudopotential parameters from the pyscf package.

    This function loads the pseudopotential parameters for an atom (given by `charge`
    argument) from the pyscf package and parses them to jnp arrays.

    Args:
        charge (int): a charge of the atom in question.
        pp_type (str): a string determining the type of pseudopotentials, it is passed
            to :func:`pyscf.gto.M()` as :data:`'ecp'` argument.
    Returns:
        tuple: a tuple containing a number of core electrons replaced by
            pseudopotential, an array of local pseudopotential parameters, and
            an array of nonlocal pseudopotential parameters.
    """
    data = next(
        iter(
            gto.M(
                atom=[
                    (int(charge), jnp.array([0, 0, 0])),
                ],
                spin=charge % 2,
                basis='6-31G',
                ecp=pp_type,
            )._ecp.values()
        )
    )
    n_core = data[0]
    pp_loc_param = data[1][0][1][1:4]
    # Pad parameters with zeros to store them in single jnp.array.
    # The fixed padding is probably not the most efficient way, (it might cause some
    # unnecessary multiplication by zero during the evaluation).
    pad = 2

    pp_loc_param = jnp.array(
        [one_param + [[0, 0]] * (pad - len(one_param)) for one_param in pp_loc_param]
    )
    pp_loc_param = jnp.swapaxes(
        pp_loc_param, -1, -2
    )  # Shape: (r^n term, coefficient (β) & exponent (α), no. of terms with the same n)

    pp_nl_coef = []
    for i in range(len(data[1]) - 1):
        # creates a list of non-local parameters; its length is the number of projectors
        pp_nl_coef.append(jnp.swapaxes(jnp.array(data[1][i + 1][1][2]), -1, -2))
    pp_nl_param = jnp.asarray(pp_nl_coef)
    return n_core, pp_loc_param, pp_nl_param
