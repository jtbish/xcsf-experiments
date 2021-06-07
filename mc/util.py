import itertools

import numpy as np


def compute_dims_ref_points(obs_space, num_ref_points_per_dim):
    dims_ref_points = []
    for dim in obs_space:
        dims_ref_points.append(
            np.linspace(dim.lower,
                        dim.upper,
                        num=num_ref_points_per_dim,
                        endpoint=True))
    return dims_ref_points


def calc_discrete_states(num_bins_per_dim, num_dims, dims_ref_points):
    # each discrete state is idenified by n-tuple giving idxs of bins
    # along dim axes, along with a "representative" continuous state that is
    # the centroid of the bin: this repr. state is used when querying the
    # actual transition function in continuous space
    possible_bin_idxs_for_dim = list(range(0, num_bins_per_dim))
    all_idx_combos = list(
        itertools.product(possible_bin_idxs_for_dim, repeat=num_dims))
    discrete_states = {}
    for idx_combo in all_idx_combos:
        repr_real_state = []
        assert len(idx_combo) == len(dims_ref_points)
        for (idx_on_dim, dim_ref_points) in zip(idx_combo, dims_ref_points):
            # compute the midpoint for this bucket on this dim
            lower_ref_point = dim_ref_points[idx_on_dim]
            upper_ref_point = dim_ref_points[idx_on_dim + 1]
            midpoint = (lower_ref_point + upper_ref_point) / 2
            repr_real_state.append(midpoint)
        repr_real_state = tuple(repr_real_state)
        discrete_states[idx_combo] = repr_real_state
    return discrete_states
