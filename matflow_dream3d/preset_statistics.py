"""Functions for replicating preset statistics models as implemented in Dream.3D"""

import numpy as np


def generate_omega3_dist_from_preset(num_bins):
    """Replicating: https://github.com/BlueQuartzSoftware/DREAM3D/blob/331c97215bb358321d9f92105a9c812a81fd1c79/Source/Plugins/SyntheticBuilding/SyntheticBuildingFilters/Presets/PrimaryRolledPreset.cpp#L62"""

    alphas = []
    betas = []
    for _ in range(num_bins):
        alpha = 10.0 + np.random.random()
        beta = 1.5 + (0.5 * np.random.random())
        alphas.append(alpha)
        betas.append(beta)

    return {'alpha': alphas, 'beta': betas}


def generate_shape_dist_from_preset(num_bins, aspect_ratio, preset_type):
    """Replicating: https://github.com/BlueQuartzSoftware/DREAM3D/blob/331c97215bb358321d9f92105a9c812a81fd1c79/Source/Plugins/SyntheticBuilding/SyntheticBuildingFilters/Presets/PrimaryRolledPreset.cpp#L88"""
    alphas = []
    betas = []
    for _ in range(num_bins):

        if preset_type in ['primary_rolled', 'precipitate_rolled']:
            alpha = (1.1 + (28.9 * (1.0 / aspect_ratio))) + np.random.random()
            beta = (30 - (28.9 * (1.0 / aspect_ratio))) + np.random.random()

        elif preset_type in ['primary_equiaxed', 'precipitate_equiaxed']:
            alpha = 15.0 + np.random.random()
            beta = 1.25 + (0.5 * np.random.random())

        alphas.append(alpha)
        betas.append(beta)

    return {'alpha': alphas, 'beta': betas}


def generate_neighbour_dist_from_preset(num_bins, preset_type):
    """Replicating: https://github.com/BlueQuartzSoftware/DREAM3D/blob/331c97215bb358321d9f92105a9c812a81fd1c79/Source/Plugins/SyntheticBuilding/SyntheticBuildingFilters/Presets/PrimaryRolledPreset.cpp#L140"""
    mus = []
    sigmas = []
    middlebin = num_bins // 2
    for i in range(num_bins):

        if preset_type == 'primary_equiaxed':
            mu = np.log(14.0 + (2.0 * float(i - middlebin)))
            sigma = 0.3 + (float(middlebin - i) / float(middlebin * 10))

        elif preset_type == 'primary_rolled':
            mu = np.log(8.0 + (1.0 * float(i - middlebin)))
            sigma = 0.3 + (float(middlebin - i) / float(middlebin * 10))

        mus.append(mu)
        sigmas.append(sigma)

    return {'average': mus, 'stddev': sigmas}
