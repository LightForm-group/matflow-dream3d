import numpy as np


def quat2euler(quats, degrees=False, P=1):
    """Convert quaternions to Bunge-convention Euler angles.

    Parameters
    ----------
    quats : ndarray of shape (N, 4) of float
        Array of N row four-vectors of unit quaternions.
    degrees : bool, optional
        If True, `euler_angles` are returned in degrees, rather than radians.

    P : int, optional
        The "P" constant, either +1 or -1, as defined within [1].

    Returns
    -------
    euler_angles : ndarray of shape (N, 3) of float
        Array of N row three-vectors of Euler angles, specified as proper Euler angles in
        the Bunge convention (rotations are about Z, new X, new new Z).    

    Notes
    -----
    Conversion of quaternions to Bunge Euler angles due to Ref. [1].

    References
    ----------
    [1] Rowenhorst, D, A D Rollett, G S Rohrer, M Groeber, M Jackson,
        P J Konijnenberg, and M De Graef. "Consistent Representations
        of and Conversions between 3D Rotations". Modelling and Simulation
        in Materials Science and Engineering 23, no. 8 (1 December 2015):
        083501. https://doi.org/10.1088/0965-0393/23/8/083501.

    """

    num_oris = quats.shape[0]
    euler_angles = np.zeros((num_oris, 3))

    q0, q1, q2, q3 = quats.T

    q03 = q0**2 + q3**2
    q12 = q1**2 + q2**2
    chi = np.sqrt(q03 * q12)

    chi_zero_idx = np.isclose(chi, 0)
    q12_zero_idx = np.isclose(q12, 0)
    q03_zero_idx = np.isclose(q03, 0)

    # Three cases are distinguished:
    idx_A = np.logical_and(chi_zero_idx, q12_zero_idx)
    idx_B = np.logical_and(chi_zero_idx, q03_zero_idx)
    idx_C = np.logical_not(chi_zero_idx)

    q0A, q3A = q0[idx_A], q3[idx_A]
    q1B, q2B = q1[idx_B], q2[idx_B]
    q0C, q1C, q2C, q3C, chiC = q0[idx_C], q1[idx_C], q2[idx_C], q3[idx_C], chi[idx_C]

    q03C = q03[idx_C]
    q12C = q12[idx_C]

    # Case 1
    euler_angles[idx_A, 0] = np.arctan2(-2 * P * q0A * q3A, q0A**2 - q3A**2)

    # Case 2
    euler_angles[idx_B, 0] = np.arctan2(2 * q1B * q2B, q1B**2 - q2B**2)
    euler_angles[idx_B, 1] = np.pi

    # Case 3
    euler_angles[idx_C, 0] = np.arctan2(
        (q1C * q3C - P * q0C * q2C) / chiC,
        (-P * q0C * q1C - q2C * q3C) / chiC,
    )
    euler_angles[idx_C, 1] = np.arctan2(2 * chiC, q03C - q12C)
    euler_angles[idx_C, 2] = np.arctan2(
        (P * q0C * q2C + q1C * q3C) / chiC,
        (q2C * q3C - P * q0C * q1C) / chiC,
    )

    euler_angles[euler_angles[:, 0] < 0, 0] += 2 * np.pi
    euler_angles[euler_angles[:, 2] < 0, 2] += 2 * np.pi

    if degrees:
        euler_angles = np.rad2deg(euler_angles)

    return euler_angles


def process_dream3D_euler_angles(euler_angles, degrees=False):
    orientations = {
        'type': 'euler',
        'euler_degrees': degrees,
        'euler_angles': euler_angles,
        'unit_cell_alignment': {'x': 'a'},
    }
    return orientations
