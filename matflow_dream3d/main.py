'`matflow_dream3d.main.py`'

import json
import copy
import warnings
from os import pread
from pathlib import Path
from textwrap import dedent

import h5py
import numpy as np
from damask_parse.utils import validate_orientations, validate_volume_element
from damask_parse.quats import axang2quat, multiply_quaternions

from matflow_dream3d import input_mapper, output_mapper
from matflow_dream3d.utilities import quat2euler, process_dream3D_euler_angles
from matflow_dream3d.preset_statistics import (
    generate_omega3_dist_from_preset,
    generate_shape_dist_from_preset,
    generate_neighbour_dist_from_preset,
)


@output_mapper(
    output_name='volume_element',
    task='segment_grains',
    method='burn',
)
def parse_dream_3D_volume_element_segmentation(path):

    with h5py.File(path, mode='r') as fh:

        container = fh['DataContainers']['DataContainer']
        grid_size = container['_SIMPL_GEOMETRY']['DIMENSIONS'][()]
        resolution = container['_SIMPL_GEOMETRY']['SPACING'][()]
        size = [i * j for i, j in zip(resolution, grid_size)]

        # make zero-indexed:
        # (not sure why FeatureIds is 4D?)
        element_material_idx = container['CellData']['FeatureIds'][()][..., 0] - 1
        element_material_idx = element_material_idx.transpose((2, 1, 0))  # reshape

        num_grains = element_material_idx.max() + 1
        phase_names = container['CellEnsembleData']['PhaseName'][()][1:]
        constituent_phase_idx = container['Grain Data']['Phases'][()][1:] - 1
        constituent_phase_label = [phase_names[i][0].decode()
                                   for i in constituent_phase_idx]
        eulers = container['Grain Data']['EulerAngles'][()][1:]

    vol_elem = {
        'grid_size': grid_size,
        'size': size,
        'element_material_idx': element_material_idx,
        'constituent_material_idx': np.arange(num_grains),
        'constituent_phase_label': constituent_phase_label,
        'material_homog': ['SX'] * num_grains,
        'orientations': process_dream3D_euler_angles(eulers),
    }
    vol_elem = validate_volume_element(vol_elem)
    return vol_elem


@output_mapper(
    output_name='volume_element',
    task='generate_volume_element',
    method='from_statistics',
)
@output_mapper(
    output_name='volume_element',
    task='generate_volume_element',
    method='from_statistics_old',
)
def parse_dream_3D_volume_element_from_stats(path):
    # TODO: check this works...

    with h5py.File(path, mode='r') as fh:
        synth_vol = fh['DataContainers']['SyntheticVolumeDataContainer']
        grid_size = synth_vol['_SIMPL_GEOMETRY']['DIMENSIONS'][()]
        resolution = synth_vol['_SIMPL_GEOMETRY']['SPACING'][()]
        size = [i * j for i, j in zip(resolution, grid_size)]

        # make zero-indexed:
        # (not sure why FeatureIds is 4D?)
        element_material_idx = synth_vol['CellData']['FeatureIds'][()][..., 0] - 1
        element_material_idx = element_material_idx.transpose((2, 1, 0))

        num_grains = element_material_idx.max() + 1
        phase_names = synth_vol['CellEnsembleData']['PhaseName'][()][1:]
        constituent_phase_idx = synth_vol['Grain Data']['Phases'][()][1:] - 1
        constituent_phase_label = [phase_names[i][0].decode()
                                   for i in constituent_phase_idx]
        eulers = synth_vol['Grain Data']['EulerAngles'][()][1:]

    vol_elem = {
        'grid_size': grid_size,
        'size': size,
        'element_material_idx': element_material_idx,
        'constituent_material_idx': np.arange(num_grains),
        'constituent_phase_label': constituent_phase_label,
        'material_homog': ['SX'] * num_grains,
        'orientations': process_dream3D_euler_angles(eulers),
    }
    vol_elem = validate_volume_element(vol_elem)
    return vol_elem


@output_mapper(
    output_name='volume_element',
    task='generate_volume_element',
    method='from_statistics_dual_phase_orientations',
)
def parse_dream_3D_volume_element_from_stats(
    path,
    phase_statistics, 
    orientations_phase_1,
    orientations_phase_2,
    RNG_seed,
):

    # TODO: make it work.
    print(f"phase_statistics: {phase_statistics}")
    print(f'ori phase 1: {orientations_phase_1}')
    print(f'ori phase 2: {orientations_phase_2}')

    with h5py.File(path, mode='r') as fh:
        synth_vol = fh['DataContainers']['SyntheticVolumeDataContainer']
        grid_size = synth_vol['_SIMPL_GEOMETRY']['DIMENSIONS'][()]
        resolution = synth_vol['_SIMPL_GEOMETRY']['SPACING'][()]
        size = [i * j for i, j in zip(resolution, grid_size)]

        # make zero-indexed:
        # (not sure why FeatureIds is 4D?)
        element_material_idx = synth_vol['CellData']['FeatureIds'][()][..., 0] - 1
        element_material_idx = element_material_idx.transpose((2, 1, 0))

        num_grains = element_material_idx.max() + 1
        phase_names = synth_vol['CellEnsembleData']['PhaseName'][()][1:]
        constituent_phase_idx = synth_vol['Grain Data']['Phases'][()][1:] - 1
        constituent_phase_label = np.array([
            phase_names[i][0].decode() for i in constituent_phase_idx
        ])
    
    ori_1 = validate_orientations(orientations_phase_1)
    ori_2 = validate_orientations(orientations_phase_2)
    oris = copy.deepcopy(ori_1) # combined orientations
    
    phase_labels = [i['name'] for i in phase_statistics]    
    phase_labels_idx = np.ones(constituent_phase_label.size) * np.nan
    for idx, i in enumerate(phase_labels):
        phase_labels_idx[constituent_phase_label == i] = idx
    assert not np.any(np.isnan(phase_labels_idx))
    phase_labels_idx = phase_labels_idx.astype(int)

    _, counts = np.unique(phase_labels_idx, return_counts=True)
    
    num_ori_1 = ori_1['quaternions'].shape[0]
    num_ori_2 = ori_2['quaternions'].shape[0]
    sampled_oris_1 = ori_1['quaternions']
    sampled_oris_2 = ori_2['quaternions']

    rng = np.random.default_rng(seed=RNG_seed)

    # If there are more orientations than phase label assignments, choose a random subset:
    if num_ori_1 != counts[0]:
        try:
            ori_1_idx = rng.choice(a=num_ori_1, size=counts[0], replace=False)
        except ValueError as err:
            raise ValueError(
                f"Probably an insufficient number of `orientations_phase_1` "
                f"({num_ori_1} given for phase {phase_labels[0]!r}, whereas {counts[0]} "
                f"needed). Caught ValueError is: {err}"
            )
        sampled_oris_1 = sampled_oris_1[ori_1_idx]
    if num_ori_2 != counts[1]:
        try:
            ori_2_idx = rng.choice(a=num_ori_2, size=counts[1], replace=False)
        except ValueError as err:
            raise ValueError(
                f"Probably an insufficient number of `orientations_phase_2` "
                f"({num_ori_2} given for phase {phase_labels[1]!r}, whereas {counts[1]} "
                f"needed). Caught ValueError is: {err}"
            )
        sampled_oris_2 = sampled_oris_2[ori_2_idx]

    ori_idx = np.ones(num_grains) * np.nan
    for idx, i in enumerate(counts):
        ori_idx[phase_labels_idx == idx] = np.arange(i) + np.sum(counts[:idx])

    if np.any(np.isnan(ori_idx)):
        raise RuntimeError("Not all phases have an orientation assigned!")
    ori_idx = ori_idx.astype(int)

    oris['quaternions'] = np.vstack([sampled_oris_1, sampled_oris_2])

    volume_element = {
        'size': size,
        'grid_size': grid_size,
        'orientations': oris,
        'element_material_idx': element_material_idx,
        'constituent_material_idx': np.arange(num_grains),
        'constituent_material_fraction': np.ones(num_grains),
        'constituent_phase_label': constituent_phase_label,
        'constituent_orientation_idx': ori_idx,
        'material_homog': np.full(num_grains, 'SX'),
    }
    volume_element = validate_volume_element(volume_element)
    return volume_element



@input_mapper(
    input_file='orientation_data.txt',
    task='segment_grains',
    method='burn',
)
def write_segment_grains_orientations_file(path, volume_element_response, increment):

    phase = volume_element_response['field_data']['phase']['data']
    ori_data = volume_element_response['field_data']['O']['data']['quaternions']

    # problem with reshaping here?
    oris_flat_SV = ori_data[increment].reshape(-1, 4, order='F')

    # Convert to vector-scalar convention used by Dream3D:
    oris_flat_VS = np.roll(oris_flat_SV, -1, axis=1)

    phase_flat = phase.reshape(-1, 1, order='F')
    phase_flat += 1  # Phases are 1-indexed in Dream3D

    col_names = [
        'Phase',
        'Quat1',
        'Quat2',
        'Quat3',
        'Quat4',
    ]

    all_dat = np.hstack([phase_flat, oris_flat_VS])
    header = ', '.join([f'{i}' for i in col_names])
    np.savetxt(
        fname=path,
        header=header,
        X=all_dat,
        fmt=['%d'] + ['%20.17f'] * (len(col_names) - 1),
        comments='',
    )


@input_mapper(
    input_file='ensemble_data.txt',
    task='segment_grains',
    method='burn',
)
def write_segment_grains_ensemble_data(path):
    # TODO: make crystal structures and phase types variables. A mapping should be
    # defineable that maps the DAMASK phases to crystalstructure and phasetype as
    # understood by Dream3D
    ensemble_data = """    [EnsembleInfo]
    Number_Phases=2

    [1]
    CrystalStructure=Cubic_High
    PhaseType=PrimaryPhase

    [2]
    CrystalStructure=Hexagonal_High
    PhaseType=PrecipitatePhase
    """
    with Path(path).open('w') as handle:
        handle.write(dedent(ensemble_data))


@input_mapper(
    input_file='pipeline.json',
    task='segment_grains',
    method='burn',
)
def write_segment_grains_pipeline(path, volume_element, misorientation_tolerance_deg):

    grid_size = [int(i) for i in volume_element['grid_size']]
    origin = [float(i) for i in volume_element.get('origin', [0, 0, 0])]
    size = [float(i) for i in volume_element.get('size', [1, 1, 1])]
    resolution = [i / j for i, j in zip(size, grid_size)]

    pipeline = {
        "0": {
            "DataContainerName": "DataContainer",
            "FilterVersion": "1.2.815",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Create Data Container",
            "Filter_Name": "CreateDataContainer",
            "Filter_Uuid": "{816fbe6b-7c38-581b-b149-3f839fb65b93}"
        },
        "1": {
            "ArrayHandling": 0,
            "BoxDimensions": (
                f"Extents:\nX Extent: 0 to {grid_size[0] - 1} (dimension: {grid_size[0]})\n"
                f"Y Extent: 0 to {grid_size[1] - 1} (dimension: {grid_size[1]})\n"
                f"Z Extent: 0 to {grid_size[2] - 1} (dimension: {grid_size[2]})\n"
                f"Bounds:\nX Range: 0.0 to 1.0 (delta: 1)\n"
                f"Y Range: 0.0 to 1.0 (delta: 1)\n"
                f"Z Range: 0.0 to 1.0 (delta: 1)\n"
            ),
            "DataContainerName": "DataContainer",
            "Dimensions": {
                "x": grid_size[0],
                "y": grid_size[1],
                "z": grid_size[2]
            },
            "EdgeAttributeMatrixName": "EdgeData",
            "FaceAttributeMatrixName0": "FaceData",
            "FaceAttributeMatrixName1": "FaceData",
            "FilterVersion": "1.2.815",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Create Geometry",
            "Filter_Name": "CreateGeometry",
            "Filter_Uuid": "{9ac220b9-14f9-581a-9bac-5714467589cc}",
            "GeometryType": 0,
            "HexCellAttributeMatrixName": "CellData",
            "ImageCellAttributeMatrixName": "CellData",
            "Origin": {
                "x": origin[0],
                "y": origin[1],
                "z": origin[2]
            },
            "RectGridCellAttributeMatrixName": "CellData",
            "Resolution": {
                "x": resolution[0],
                "y": resolution[1],
                "z": resolution[2]
            },
            "SharedEdgeListArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "SharedHexListArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "SharedQuadListArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "SharedTetListArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "SharedTriListArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "SharedVertexListArrayPath0": {
                "Attribute Matrix Name": "CellData2",
                "Data Array Name": "coords",
                "Data Container Name": "DataContainer"
            },
            "SharedVertexListArrayPath1": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "SharedVertexListArrayPath2": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "SharedVertexListArrayPath3": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "SharedVertexListArrayPath4": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "SharedVertexListArrayPath5": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "TetCellAttributeMatrixName": "CellData",
            "TreatWarningsAsErrors": 0,
            "VertexAttributeMatrixName0": "VertexData",
            "VertexAttributeMatrixName1": "VertexData",
            "VertexAttributeMatrixName2": "VertexData",
            "VertexAttributeMatrixName3": "VertexData",
            "VertexAttributeMatrixName4": "VertexData",
            "VertexAttributeMatrixName5": "VertexData",
            "XBoundsArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "YBoundsArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "ZBoundsArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            }
        },
        "2": {
            "FilterVersion": "1.2.815",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Import ASCII Data",
            "Filter_Name": "ReadASCIIData",
            "Filter_Uuid": "{bdb978bc-96bf-5498-972c-b509c38b8d50}",
            "Wizard_AttributeMatrixType": 3,
            "Wizard_AutomaticAM": True,
            "Wizard_BeginIndex": 2,
            "Wizard_ConsecutiveDelimiters": 1,
            "Wizard_DataHeaders": [
                "Phase",
                "Quat1",
                "Quat2",
                "Quat3",
                "Quat4"
            ],
            "Wizard_DataTypes": [
                "int32_t",
                "float",
                "float",
                "float",
                "float"
            ],
            "Wizard_Delimiters": ", ",
            "Wizard_HeaderIsCustom": True,
            "Wizard_HeaderLine": -1,
            "Wizard_HeaderUseDefaults": False,
            "Wizard_InputFilePath": str(Path(path).parent.joinpath('orientation_data.txt')),
            "Wizard_NumberOfLines": int(np.product(grid_size) + 1),
            "Wizard_SelectedPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "",
                "Data Container Name": "DataContainer"
            },
            "Wizard_TupleDims": [
                grid_size[0],
                grid_size[1],
                grid_size[2]
            ]
        },
        "3": {
            "FilterVersion": "1.2.815",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Combine Attribute Arrays",
            "Filter_Name": "CombineAttributeArrays",
            "Filter_Uuid": "{a6b50fb0-eb7c-5d9b-9691-825d6a4fe772}",
            "MoveValues": 1,
            "NormalizeData": 0,
            "SelectedDataArrayPaths": [
                {
                    "Attribute Matrix Name": "CellData",
                    "Data Array Name": "Quat1",
                    "Data Container Name": "DataContainer"
                },
                {
                    "Attribute Matrix Name": "CellData",
                    "Data Array Name": "Quat2",
                    "Data Container Name": "DataContainer"
                },
                {
                    "Attribute Matrix Name": "CellData",
                    "Data Array Name": "Quat3",
                    "Data Container Name": "DataContainer"
                },
                {
                    "Attribute Matrix Name": "CellData",
                    "Data Array Name": "Quat4",
                    "Data Container Name": "DataContainer"
                }
            ],
            "StackedDataArrayName": "quats"
        },
        "4": {
            "CellEnsembleAttributeMatrixName": "EnsembleAttributeMatrix",
            "CrystalStructuresArrayName": "CrystalStructures",
            "DataContainerName": "DataContainer",
            "FilterVersion": "6.5.141",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Import Ensemble Info File",
            "Filter_Name": "EnsembleInfoReader",
            "Filter_Uuid": "{33a37a47-d002-5c18-b270-86025881fe1e}",
            "InputFile": str(Path(path).parent.joinpath('ensemble_data.txt')),
            "PhaseTypesArrayName": "PhaseTypes"
        },
        "5": {
            "ActiveArrayName": "Active",
            "CellFeatureAttributeMatrixName": "CellFeatureData",
            "CellPhasesArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "Phase",
                "Data Container Name": "DataContainer"
            },
            "CrystalStructuresArrayPath": {
                "Attribute Matrix Name": "EnsembleAttributeMatrix",
                "Data Array Name": "CrystalStructures",
                "Data Container Name": "DataContainer"
            },
            "FeatureIdsArrayName": "FeatureIds",
            "FilterVersion": "6.5.141",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Segment Features (Misorientation)",
            "Filter_Name": "EBSDSegmentFeatures",
            "Filter_Uuid": "{7861c691-b821-537b-bd25-dc195578e0ea}",
            "GoodVoxelsArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "Mask",
                "Data Container Name": "ImageDataContainer"
            },
            "MisorientationTolerance": misorientation_tolerance_deg,
            "QuatsArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "quats",
                "Data Container Name": "DataContainer"
            },
            "UseGoodVoxels": 0
        },
        "6": {
            "FilterVersion": "1.2.815",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Write DREAM.3D Data File",
            "Filter_Name": "DataContainerWriter",
            "Filter_Uuid": "{3fcd4c43-9d75-5b86-aad4-4441bc914f37}",
            "OutputFile": f"{str(Path(path).parent.joinpath('pipeline.dream3d'))}",
            "WriteTimeSeries": 0,
            "WriteXdmfFile": 1
        },
        "PipelineBuilder": {
            "Name": "segment_features",
            "Number_Filters": 7,
            "Version": 6
        }
    }
    with Path(path).open('w') as fh:
        json.dump(pipeline, fh, indent=4)


@input_mapper(
    input_file='pipeline.json',
    task='generate_volume_element',
    method='from_statistics_old',
)
def generate_RVE_pipeline(path, grid_size, resolution, size, origin, periodic):

    if resolution is None:
        resolution = [i / j for i, j in zip(size, grid_size)]

    if origin is None:
        origin = [0, 0, 0]

    pipeline = {
        "0": {
            "CellEnsembleAttributeMatrixName": "CellEnsembleData",
            "CrystalStructuresArrayName": "CrystalStructures",
            "Filter_Human_Label": "StatsGenerator",
            "Filter_Name": "StatsGeneratorFilter",
            "PhaseTypesArrayName": "PhaseTypes",
            "StatsDataArray": {
                "1": {
                    "AxisODF-Weights": {
                    },
                    "Bin Count": 4,
                    "BinNumber": [
                        7.3890562057495117,
                        17.389057159423828,
                        27.389057159423828,
                        37.389057159423828
                    ],
                    "BoundaryArea": 0,
                    "Crystal Symmetry": 1,
                    "FeatureSize Distribution": {
                        "Average": 3,
                        "Standard Deviation": 0.25
                    },
                    "FeatureSize Vs B Over A Distributions": {
                        "Alpha": [
                            15.845513343811035,
                            15.281289100646973,
                            15.406131744384766,
                            15.695631980895996
                        ],
                        "Beta": [
                            1.5363599061965942,
                            1.3575199842453003,
                            1.2908644676208496,
                            1.6510697603225708
                        ],
                        "Distribution Type": "Beta Distribution"
                    },
                    "FeatureSize Vs C Over A Distributions": {
                        "Alpha": [
                            15.830905914306641,
                            15.119057655334473,
                            15.210259437561035,
                            15.403964042663574
                        ],
                        "Beta": [
                            1.4798208475112915,
                            1.4391646385192871,
                            1.6361048221588135,
                            1.3149876594543457
                        ],
                        "Distribution Type": "Beta Distribution"
                    },
                    "FeatureSize Vs Neighbors Distributions": {
                        "Average": [
                            2.3025851249694824,
                            2.4849066734313965,
                            2.6390573978424072,
                            2.7725887298583984
                        ],
                        "Distribution Type": "Log Normal Distribution",
                        "Standard Deviation": [
                            0.40000000596046448,
                            0.34999999403953552,
                            0.30000001192092896,
                            0.25
                        ]
                    },
                    "FeatureSize Vs Omega3 Distributions": {
                        "Alpha": [
                            10.906224250793457,
                            10.030556678771973,
                            10.367804527282715,
                            10.777519226074219
                        ],
                        "Beta": [
                            1.7305665016174316,
                            1.6383645534515381,
                            1.6687047481536865,
                            1.6839183568954468
                        ],
                        "Distribution Type": "Beta Distribution"
                    },
                    "Feature_Diameter_Info": [
                        10,
                        42.521083831787109,
                        7.3890562057495117
                    ],
                    "MDF-Weights": {
                    },
                    "Name": "Primary",
                    "ODF-Weights": {
                    },
                    "PhaseFraction": 0.89999997615814209,
                    "PhaseType": "Primary"
                },
                "2": {
                    "AxisODF-Weights": {
                    },
                    "Bin Count": 4,
                    "BinNumber": [
                        2.2255408763885498,
                        4.2255411148071289,
                        6.2255411148071289,
                        8.2255411148071289
                    ],
                    "BoundaryArea": 64498012,
                    "Crystal Symmetry": 0,
                    "FeatureSize Distribution": {
                        "Average": 1.6000000238418579,
                        "Standard Deviation": 0.20000000298023224
                    },
                    "FeatureSize Vs B Over A Distributions": {
                        "Alpha": [
                            15.258569717407227,
                            15.15038013458252,
                            15.949015617370605,
                            15.441672325134277
                        ],
                        "Beta": [
                            1.6226730346679688,
                            1.5978513956069946,
                            1.4994683265686035,
                            1.5526076555252075
                        ],
                        "Distribution Type": "Beta Distribution"
                    },
                    "FeatureSize Vs C Over A Distributions": {
                        "Alpha": [
                            15.780433654785156,
                            15.858841896057129,
                            15.259775161743164,
                            15.857120513916016
                        ],
                        "Beta": [
                            1.5344709157943726,
                            1.2825722694396973,
                            1.649916410446167,
                            1.7178913354873657
                        ],
                        "Distribution Type": "Beta Distribution"
                    },
                    "FeatureSize Vs Omega3 Distributions": {
                        "Alpha": [
                            10.484344482421875,
                            10.260377883911133,
                            10.586400985717773,
                            10.218396186828613
                        ],
                        "Beta": [
                            1.5603832006454468,
                            1.599597692489624,
                            1.5324842929840088,
                            1.5695462226867676
                        ],
                        "Distribution Type": "Beta Distribution"
                    },
                    "Feature_Diameter_Info": [
                        2,
                        9.0250139236450195,
                        2.2255408763885498
                    ],
                    "MDF-Weights": {
                    },
                    "Name": "Precipitate",
                    "ODF-Weights": {
                    },
                    "PhaseFraction": 0.10000000149011612,
                    "PhaseType": "Precipitate",
                    "Precipitate Boundary Fraction": 0.69999998807907104,
                    "Radial Distribution Function": {
                        "Bin Count": 50,
                        "BoxDims": [
                            100,
                            100,
                            100
                        ],
                        "BoxRes": [
                            0.10000000149011612,
                            0.10000000149011612,
                            0.10000000149011612
                        ],
                        "Max": 80,
                        "Min": 10
                    }
                },
                "Name": "Statistics",
                "Phase Count": 3
            },
            "StatsDataArrayName": "Statistics",
            "StatsGeneratorDataContainerName": "StatsGeneratorDataContainer"
        },
        "1": {
            "CellAttributeMatrixName": "CellData",
            "DataContainerName": "SyntheticVolumeDataContainer",
            "Dimensions": {
                "x": grid_size[0],
                "y": grid_size[1],
                "z": grid_size[2],
            },
            "EstimateNumberOfFeatures": 0,
            "FilterVersion": "1.0.278",
            "Filter_Human_Label": "Initialize Synthetic Volume",
            "Filter_Name": "InitializeSyntheticVolume",
            "InputPhaseTypesArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "PhaseTypes",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "InputStatsArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "Statistics",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "InputStatsFile": "",
            "Origin": {
                "x": origin[0],
                "y": origin[1],
                "z": origin[2],
            },
            "Resolution": {
                "x": resolution[0],
                "y": resolution[1],
                "z": resolution[2],
            }
        },
        "2": {
            "FilterVersion": "1.0.278",
            "Filter_Human_Label": "Establish Shape Types",
            "Filter_Name": "EstablishShapeTypes",
            "InputPhaseTypesArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "PhaseTypes",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "ShapeTypeData": [
                999,
                0,
                0
            ],
            "ShapeTypesArrayName": "ShapeTypes"
        },
        "3": {
            "CellPhasesArrayName": "Phases",
            "CsvOutputFile": "",
            "ErrorOutputFile": "",
            "FeatureIdsArrayName": "FeatureIds",
            "FeatureInputFile": "",
            "FeaturePhasesArrayName": "Phases",
            "FilterVersion": "1.0.278",
            "Filter_Human_Label": "Pack Primary Phases",
            "Filter_Name": "PackPrimaryPhases",
            "FeatureGeneration": 0,
            "InputPhaseTypesArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "PhaseTypes",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "InputShapeTypesArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "ShapeTypes",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "InputStatsArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "Statistics",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "MaskArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "NumFeaturesArrayName": "NumFeatures",
            "OutputCellAttributeMatrixPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "OutputCellEnsembleAttributeMatrixName": "CellEnsembleData",
            "OutputCellFeatureAttributeMatrixName": "Grain Data",
            "PeriodicBoundaries": int(periodic),
            "UseMask": 0,
            "VtkOutputFile": "",
            "WriteGoalAttributes": 0
        },
        "4": {
            "BoundaryCellsArrayName": "BoundaryCells",
            "FeatureIdsArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "FeatureIds",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "FilterVersion": "1.0.278",
            "Filter_Human_Label": "Find Boundary Cells (Image)",
            "Filter_Name": "FindBoundaryCells"
        },
        "5": {
            "BoundaryCellsArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "BoundaryCells",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "CellPhasesArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "Phases",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "CsvOutputFile": "",
            "FeatureIdsArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "FeatureIds",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "FeaturePhasesArrayPath": {
                "Attribute Matrix Name": "Grain Data",
                "Data Array Name": "Phases",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "FilterVersion": "1.0.278",
            "Filter_Human_Label": "Insert Precipitate Phases",
            "Filter_Name": "InsertPrecipitatePhases",
            "HavePrecips": 0,
            "InputPhaseTypesArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "PhaseTypes",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "InputShapeTypesArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "ShapeTypes",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "InputStatsArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "Statistics",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "MaskArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "MatchRDF": 0,
            "NumFeaturesArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "NumFeatures",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "PeriodicBoundaries": int(periodic),
            "PrecipInputFile": "",
            "UseMask": 0,
            "WriteGoalAttributes": 0
        },
        "6": {
            "BoundaryCellsArrayName": "BoundaryCells",
            "CellFeatureAttributeMatrixPath": {
                "Attribute Matrix Name": "Grain Data",
                "Data Array Name": "",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "FeatureIdsArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "FeatureIds",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "FilterVersion": "1.0.278",
            "Filter_Human_Label": "Find Feature Neighbors",
            "Filter_Name": "FindNeighbors",
            "NeighborListArrayName": "NeighborList",
            "NumNeighborsArrayName": "NumNeighbors",
            "SharedSurfaceAreaListArrayName": "SharedSurfaceAreaList",
            "StoreBoundaryCells": 0,
            "StoreSurfaceFeatures": 1,
            "SurfaceFeaturesArrayName": "SurfaceFeatures"
        },
        "7": {
            "AvgQuatsArrayName": "AvgQuats",
            "CellEulerAnglesArrayName": "EulerAngles",
            "CrystalStructuresArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "CrystalStructures",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "FeatureEulerAnglesArrayName": "EulerAngles",
            "FeatureIdsArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "FeatureIds",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "FeaturePhasesArrayPath": {
                "Attribute Matrix Name": "Grain Data",
                "Data Array Name": "Phases",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "FilterVersion": "1.0.278",
            "Filter_Human_Label": "Match Crystallography",
            "Filter_Name": "MatchCrystallography",
            "InputStatsArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "Statistics",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "MaxIterations": 100000,
            "NeighborListArrayPath": {
                "Attribute Matrix Name": "Grain Data",
                "Data Array Name": "NeighborList",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "NumFeaturesArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "NumFeatures",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "PhaseTypesArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "PhaseTypes",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "SharedSurfaceAreaListArrayPath": {
                "Attribute Matrix Name": "Grain Data",
                "Data Array Name": "SharedSurfaceAreaList",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "SurfaceFeaturesArrayPath": {
                "Attribute Matrix Name": "Grain Data",
                "Data Array Name": "SurfaceFeatures",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "VolumesArrayName": "Volumes"
        },
        "8": {
            "CellEulerAnglesArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "EulerAngles",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "CellIPFColorsArrayName": "IPFColor",
            "CellPhasesArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "Phases",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "CrystalStructuresArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "CrystalStructures",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "FilterVersion": "1.0.278",
            "Filter_Human_Label": "Generate IPF Colors",
            "Filter_Name": "GenerateIPFColors",
            "GoodVoxelsArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "ReferenceDir": {
                "x": 0,
                "y": 0,
                "z": 1
            },
            "UseGoodVoxels": 0
        },
        "9": {
            "FilterVersion": "1.0.278",
            "Filter_Human_Label": "Write DREAM.3D Data File",
            "Filter_Name": "DataContainerWriter",
            "OutputFile": f"{str(Path(path).parent.joinpath('pipeline.dream3d'))}",
            "WriteXdmfFile": 1
        },
        "PipelineBuilder": {
            "Name": "(04) Two Phase Cubic Hexagonal Particles Equiaxed",
            "Number_Filters": 10,
            "Version": "1.0"
        }
    }

    with Path(path).open('w') as fh:
        json.dump(pipeline, fh, indent=4)


@input_mapper(
    input_file='volume_element.txt',
    task='visualise_volume_element',
    method='Dream3D',
)
def write_visualise_volume_element_text_file(path, volume_element):

    header = 'GrainID'
    np.savetxt(
        fname=path,
        header=header,
        X=volume_element['element_material_idx'].flatten(order='F'),
        fmt='%d',
        comments='',
    )


@input_mapper(
    input_file='pipeline.json',
    task='visualise_volume_element',
    method='Dream3D',
)
def write_visualise_volume_element_pipeline(path, volume_element):

    grid_size = [int(i) for i in volume_element['grid_size']]
    origin = [float(i) for i in volume_element.get('origin', [0, 0, 0])]
    size = [float(i) for i in volume_element.get('size', [1, 1, 1])]
    resolution = [i / j for i, j in zip(size, grid_size)]

    pipeline = {
        "0": {
            "DataContainerName": "DataContainer",
            "FilterVersion": "1.2.815",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Create Data Container",
            "Filter_Name": "CreateDataContainer",
            "Filter_Uuid": "{816fbe6b-7c38-581b-b149-3f839fb65b93}"
        },
        "1": {
            "ArrayHandling": 0,
            "BoxDimensions": (
                f"Extents:\nX Extent: 0 to {grid_size[0] - 1} (dimension: {grid_size[0]})\n"
                f"Y Extent: 0 to {grid_size[1] - 1} (dimension: {grid_size[1]})\n"
                f"Z Extent: 0 to {grid_size[2] - 1} (dimension: {grid_size[2]})\n"
                f"Bounds:\nX Range: 0.0 to 1.0 (delta: 1)\n"
                f"Y Range: 0.0 to 1.0 (delta: 1)\n"
                f"Z Range: 0.0 to 1.0 (delta: 1)\n"
            ),
            "DataContainerName": "DataContainer",
            "Dimensions": {
                "x": grid_size[0],
                "y": grid_size[1],
                "z": grid_size[2]
            },
            "EdgeAttributeMatrixName": "EdgeData",
            "FaceAttributeMatrixName0": "FaceData",
            "FaceAttributeMatrixName1": "FaceData",
            "FilterVersion": "1.2.815",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Create Geometry",
            "Filter_Name": "CreateGeometry",
            "Filter_Uuid": "{9ac220b9-14f9-581a-9bac-5714467589cc}",
            "GeometryType": 0,
            "HexCellAttributeMatrixName": "CellData",
            "ImageCellAttributeMatrixName": "CellData",
            "Origin": {
                "x": origin[0],
                "y": origin[1],
                "z": origin[2]
            },
            "RectGridCellAttributeMatrixName": "CellData",
            "Resolution": {
                "x": resolution[0],
                "y": resolution[1],
                "z": resolution[2]
            },
            "SharedEdgeListArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "SharedHexListArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "SharedQuadListArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "SharedTetListArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "SharedTriListArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "SharedVertexListArrayPath0": {
                "Attribute Matrix Name": "CellData2",
                "Data Array Name": "coords",
                "Data Container Name": "DataContainer"
            },
            "SharedVertexListArrayPath1": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "SharedVertexListArrayPath2": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "SharedVertexListArrayPath3": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "SharedVertexListArrayPath4": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "SharedVertexListArrayPath5": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "TetCellAttributeMatrixName": "CellData",
            "TreatWarningsAsErrors": 0,
            "VertexAttributeMatrixName0": "VertexData",
            "VertexAttributeMatrixName1": "VertexData",
            "VertexAttributeMatrixName2": "VertexData",
            "VertexAttributeMatrixName3": "VertexData",
            "VertexAttributeMatrixName4": "VertexData",
            "VertexAttributeMatrixName5": "VertexData",
            "XBoundsArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "YBoundsArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "ZBoundsArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            }
        },
        "2": {
            "FilterVersion": "1.2.815",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Import ASCII Data",
            "Filter_Name": "ReadASCIIData",
            "Filter_Uuid": "{bdb978bc-96bf-5498-972c-b509c38b8d50}",
            "Wizard_AttributeMatrixType": 3,
            "Wizard_AutomaticAM": False,
            "Wizard_BeginIndex": 2,
            "Wizard_ConsecutiveDelimiters": 0,
            "Wizard_DataHeaders": [
                "GrainID"
            ],
            "Wizard_DataTypes": [
                "int32_t"
            ],
            "Wizard_Delimiters": ", ",
            "Wizard_HeaderIsCustom": True,
            "Wizard_HeaderLine": -1,
            "Wizard_HeaderUseDefaults": False,
            "Wizard_InputFilePath": str(Path(path).parent.joinpath('volume_element.txt')),
            "Wizard_NumberOfLines": int(np.product(grid_size) + 1),
            "Wizard_SelectedPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "",
                "Data Container Name": "DataContainer"
            },
            "Wizard_TupleDims": [
                grid_size[0],
                grid_size[1],
                grid_size[2]
            ]
        },
        "3": {
            "FilterVersion": "1.2.815",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Write DREAM.3D Data File",
            "Filter_Name": "DataContainerWriter",
            "Filter_Uuid": "{3fcd4c43-9d75-5b86-aad4-4441bc914f37}",
            "OutputFile": f"{str(Path(path).parent.joinpath('pipeline.dream3d'))}",
            "WriteTimeSeries": 0,
            "WriteXdmfFile": 1
        },
        "PipelineBuilder": {
            "Name": "visualise_volume_element",
            "Number_Filters": 4,
            "Version": 6
        }
    }

    with Path(path).open('w') as fh:
        json.dump(pipeline, fh, indent=4)

@input_mapper(
    input_file="precipitates.txt",
    task="generate_volume_element",
    method="from_statistics_dual_phase_orientations"
)
@input_mapper(
    input_file="precipitates.txt",
    task="generate_volume_element",
    method="from_statistics"
)
def write_precipitate_file(path, precipitates):
    if precipitates:
        with Path(path).open('wt') as fp:
            fp.write(str(len(precipitates)) + '\n')
            for i in precipitates:
                fp.write(
                    f"{i['phase_number']} "
                    f"{i['position'][0]:.6f} {i['position'][1]:.6f} {i['position'][2]:.6f} "
                    f"{i['major_semi_axis_length']:.6f} "
                    f"{i['mid_semi_axis_length']:.6f} "
                    f"{i['minor_semi_axis_length']:.6f} "
                    f"{i.get('omega3', 1):.6f} "
                    f"{i['euler_angle'][0]:.6f} "
                    f"{i['euler_angle'][1]:.6f} "
                    f"{i['euler_angle'][2]:.6f}\n"
                )


@input_mapper(
    input_file='pipeline.json',
    task='generate_volume_element',
    method='from_statistics_dual_phase_orientations',
)
def generate_RVE_from_statistics_dual_phase_pipeline_writer(
    path,
    grid_size,
    resolution,
    size,
    origin,
    periodic,
    phase_statistics,
    precipitates,
):
    return generate_RVE_from_statistics_pipeline_writer(
        path,
        grid_size,
        resolution,
        size,
        origin,
        periodic,
        phase_statistics,
        precipitates,
        orientations=None,
    )


@input_mapper(
    input_file='pipeline.json',
    task='generate_volume_element',
    method='from_statistics',
)
def generate_RVE_from_statistics_pipeline_writer(
    path,
    grid_size,
    resolution,
    size,
    origin,
    periodic,
    phase_statistics,
    precipitates,
    orientations,
):

    # TODO: fix BoxDimensions in filter 01?
    # TODO: unsure how to set precipitate RDF BoxRes?

    if resolution is None:
        resolution = [i / j for i, j in zip(size, grid_size)]

    if origin is None:
        origin = [0, 0, 0]

    REQUIRED_PHASE_BASE_KEYS = {
        'type',
        'name',
        'crystal_structure',
        'volume_fraction',
    }
    REQUIRED_PHASE_NON_MATRIX_KEYS = REQUIRED_PHASE_BASE_KEYS | {
        'size_distribution',
    }
    REQUIRED_PHASE_KEYS = {
        'matrix': REQUIRED_PHASE_BASE_KEYS,
        'primary': REQUIRED_PHASE_NON_MATRIX_KEYS,
        'precipitate': REQUIRED_PHASE_NON_MATRIX_KEYS | {
            'radial_distribution_function',
            'number_fraction_on_boundary',
        },
    }
    ALLOWED_PHASE_NON_MATRIX_KEYS = REQUIRED_PHASE_NON_MATRIX_KEYS | {
        'preset_statistics_model',
        'ODF',
        'axis_ODF',
    }
    ALLOWED_PHASE_KEYS = {
        'matrix': REQUIRED_PHASE_KEYS['matrix'],
        'primary': REQUIRED_PHASE_KEYS['primary'] | ALLOWED_PHASE_NON_MATRIX_KEYS,
        'precipitate': REQUIRED_PHASE_KEYS['precipitate'] | ALLOWED_PHASE_NON_MATRIX_KEYS,
    }
    ALLOWED_PHASE_TYPES = set(REQUIRED_PHASE_KEYS.keys())
    REQUIRED_PHASE_SIZE_DIST_KEYS = {
        'ESD_log_stddev',
    }
    ALLOWED_PHASE_SIZE_DIST_KEYS = REQUIRED_PHASE_SIZE_DIST_KEYS | {
        'ESD_log_mean',
        'ESD_mean',
        'ESD_log_stddev_min_cut_off',
        'ESD_log_stddev_max_cut_off',
        'bin_step_size',
        'num_bins',
        'omega3',
        'b/a',
        'c/a',
        'neighbours',
    }
    ALLOWED_PRECIP_RDF_KEYS = {
        'min_distance',
        'max_distance',
        'num_bins',
        'box_size',
    }
    ALLOWED_CRYSTAL_STRUCTURES = {  # values are crystal symmetry index:
        'hexagonal': 0,
        'cubic': 1,
    }
    SIGMA_MIN_DEFAULT = 5
    SIGMA_MAX_DEFAULT = 5
    # Distributions defined for each size distribution bin:
    DISTRIBUTIONS_MAP = {
        'omega3': {
            'type': 'beta',
            'default_keys': {
                'alpha': 10.0,
                'beta': 1.5,
            },
            'label': 'FeatureSize Vs Omega3 Distributions',
        },
        'b/a': {
            'type': 'beta',
            'default_keys': {
                'alpha': 10.0,
                'beta': 1.5,
            },
            'label': 'FeatureSize Vs B Over A Distributions',
        },
        'c/a': {
            'type': 'beta',
            'default_keys': {
                'alpha': 10.0,
                'beta': 1.5,
            },
            'label': 'FeatureSize Vs C Over A Distributions',
        },
        'neighbours': {
            'type': 'lognormal',
            'default_keys': {
                'average': 2.0,
                'stddev': 0.5,
            },
            'label': 'FeatureSize Vs Neighbors Distributions',
        },
    }
    DISTRIBUTIONS_TYPE_LABELS = {
        'lognormal': 'Log Normal Distribution',
        'beta': 'Beta Distribution',
    }
    DISTRIBUTIONS_KEY_LABELS = {
        'alpha': 'Alpha',
        'beta': 'Beta',
        'average': 'Average',
        'stddev': 'Standard Deviation',
    }
    PRESETS_TYPE_KEYS = {
        'primary_equiaxed': {
            'type',
        },
        'primary_rolled': {
            'type',
            'A_axis_length',
            'B_axis_length',
            'C_axis_length',
        },
        'precipitate_equiaxed': {
            'type',
        },
        'precipitate_rolled': {
            'type',
            'A_axis_length',
            'B_axis_length',
            'C_axis_length',
        },
    }
    REQUIRED_PHASE_AXIS_ODF_KEYS = {'orientations'}
    ALLOWED_PHASE_AXIS_ODF_KEYS = REQUIRED_PHASE_AXIS_ODF_KEYS | {'weights', 'sigmas'}
    REQUIRED_PHASE_ODF_KEYS = set()  # presets can be specified instead of orientations
    ALLOWED_PHASE_ODF_KEYS = ALLOWED_PHASE_AXIS_ODF_KEYS | {'presets'}
    DEFAULT_ODF_WEIGHT = 500_000
    DEFAULT_ODF_SIGMA = 2

    DEFAULT_AXIS_ODF_WEIGHT = DEFAULT_ODF_WEIGHT
    DEFAULT_AXIS_ODF_SIGMA = DEFAULT_ODF_SIGMA

    ODF_CUBIC_PRESETS = {
        'cube': (0, 0, 0),
        'goss': (0, 45, 0),
        'brass': (35, 45, 0),
        'copper': (90, 35, 45),
        's': (59, 37, 63),
        's1': (55, 30, 65),
        's2': (45, 35, 65),
        'rc(rd1)': (0, 20, 0),
        'rc(rd2)': (0, 35, 0),
        'rc(nd1)': (20, 0, 0),
        'rc(nd2)': (35, 0, 0),
        'p': (70, 45, 0),
        'q': (55, 20, 0),
        'r': (55, 75, 25),
    }

    vol_frac_sum = 0.0
    stats_JSON = []
    for phase_idx, phase_stats in enumerate(phase_statistics):

        # Validation:

        err_msg = f'Problem with `phase_statistics` index {phase_idx}: '

        phase_type = phase_stats['type'].lower()
        if phase_type not in ALLOWED_PHASE_TYPES:
            raise ValueError(err_msg + f'`type` "{phase_stats["type"]}" not allowed.')

        given_keys = set(phase_stats.keys())
        miss_keys = REQUIRED_PHASE_KEYS[phase_type] - given_keys
        bad_keys = given_keys - ALLOWED_PHASE_KEYS[phase_type]
        if miss_keys:
            msg = err_msg + f'Missing keys: {", ".join([f"{i}" for i in miss_keys])}'
            raise ValueError(msg)
        if bad_keys:
            msg = err_msg + f'Unknown keys: {", ".join([f"{i}" for i in bad_keys])}'
            raise ValueError(msg)

        size_dist = phase_stats['size_distribution']
        given_size_keys = set(size_dist.keys())
        miss_size_keys = REQUIRED_PHASE_SIZE_DIST_KEYS - given_size_keys
        bad_size_keys = given_size_keys - ALLOWED_PHASE_SIZE_DIST_KEYS
        if miss_size_keys:
            raise ValueError(
                err_msg + f'Missing `size_distribution` keys: '
                f'{", ".join([f"{i}" for i in miss_size_keys])}'
            )
        if bad_size_keys:
            raise ValueError(
                err_msg + f'Unknown `size_distribution` keys: '
                f'{", ".join([f"{i}" for i in bad_size_keys])}'
            )
        num_bins = size_dist.get('num_bins')
        bin_step_size = size_dist.get('bin_step_size')
        if sum([i is None for i in (num_bins, bin_step_size)]) != 1:
            raise ValueError(
                err_msg + f'Specify exactly one of `num_bins` (given as "{num_bins}") '
                f'and `bin_step_size` (given as "{bin_step_size}").'
            )

        if phase_type == 'precipitate':
            RDF = phase_stats['radial_distribution_function']
            given_RDF_keys = set(RDF.keys())
            miss_RDF_keys = ALLOWED_PRECIP_RDF_KEYS - given_RDF_keys
            bad_RDF_keys = given_RDF_keys - ALLOWED_PRECIP_RDF_KEYS

            if miss_RDF_keys:
                raise ValueError(
                    err_msg + f'Missing `radial_distribution_function` keys: '
                    f'{", ".join([f"{i}" for i in miss_RDF_keys])}'
                )
            if bad_RDF_keys:
                raise ValueError(
                    err_msg + f'Unknown `radial_distribution_function` keys: '
                    f'{", ".join([f"{i}" for i in bad_RDF_keys])}'
                )

        phase_i_CS = phase_stats['crystal_structure']
        if phase_i_CS not in ALLOWED_CRYSTAL_STRUCTURES:
            msg = err_msg + (
                f'`crystal_structure` value "{phase_i_CS}" unknown. Must be one of: '
                f'{", ".join([f"{i}" for i in ALLOWED_CRYSTAL_STRUCTURES])}'
            )
            raise ValueError(msg)

        preset = phase_stats.get('preset_statistics_model')
        if preset:
            given_preset_keys = set(preset.keys())
            preset_type = preset.get('type')
            if not preset_type:
                raise ValueError(err_msg + f'Missing `preset_statistics_model` key: '
                                 f'"type".')
            miss_preset_keys = PRESETS_TYPE_KEYS[preset_type] - given_preset_keys
            bad_preset_keys = given_preset_keys - PRESETS_TYPE_KEYS[preset_type]

            if miss_preset_keys:
                raise ValueError(
                    err_msg + f'Missing `preset_statistics_model` keys: '
                    f'{", ".join([f"{i}" for i in miss_preset_keys])}'
                )
            if bad_preset_keys:
                raise ValueError(
                    err_msg + f'Unknown `preset_statistics_model` keys: '
                    f'{", ".join([f"{i}" for i in bad_preset_keys])}'
                )

            if 'rolled' in preset_type:
                # check: A >= B >= C
                if not (
                    preset['A_axis_length'] >=
                    preset['B_axis_length'] >=
                    preset['C_axis_length']
                ):
                    raise ValueError(
                        err_msg + f'The following condition must be true: '
                        f'`A_axis_length >= B_axis_length >= C_axis_length`, but these '
                        f'are, respectively: {preset["A_axis_length"]}, '
                        f'{preset["B_axis_length"]}, {preset["C_axis_length"]}.'
                    )

        # Sum given volume fractions:
        vol_frac_sum += phase_stats['volume_fraction']

        log_mean = size_dist.get('ESD_log_mean')
        mean = size_dist.get('ESD_mean')
        if sum([i is None for i in (log_mean, mean)]) != 1:
            raise ValueError(
                err_msg + f'Specify exactly one of `ESD_log_mean` (given as '
                f'"{log_mean}") and `ESD_mean` (given as "{mean}").'
            )

        sigma = size_dist['ESD_log_stddev']
        if log_mean is None:
            # expected value (mean) of the variable's natural log
            log_mean = np.log(mean) - (sigma ** 2 / 2)

        sigma_min = size_dist.get('ESD_log_stddev_min_cut_off', SIGMA_MIN_DEFAULT)
        sigma_max = size_dist.get('ESD_log_stddev_max_cut_off', SIGMA_MAX_DEFAULT)
        min_feat_ESD = np.exp(log_mean - (sigma_min * sigma))
        max_feat_ESD = np.exp(log_mean + (sigma_max * sigma))

        if bin_step_size is not None:
            bins = np.arange(min_feat_ESD, max_feat_ESD, bin_step_size)
            num_bins = len(bins)
        else:
            bin_step_size = (max_feat_ESD - min_feat_ESD) / num_bins
            bins = np.linspace(min_feat_ESD, max_feat_ESD, num_bins, endpoint=False)

        feat_diam_info = [bin_step_size, max_feat_ESD, min_feat_ESD]

        # Validate other distributions after sorting out number of bins:
        all_dists = {}
        for dist_key, dist_info in DISTRIBUTIONS_MAP.items():

            dist = size_dist.get(dist_key)
            if not dist:
                if not preset:
                    dist = copy.deepcopy(dist_info['default_keys'])
                else:
                    continue
            else:
                if dist_key == 'neighbours' and phase_type == 'precipitate':
                    warnings.warn(err_msg + f'`neighbours` distribution not allowed with '
                                  f'"precipitate" phase type; ignoring.')
                    continue

            required_dist_keys = set(dist_info['default_keys'].keys())
            given_dist_keys = set(dist.keys())

            miss_dist_keys = required_dist_keys - given_dist_keys
            bad_dist_keys = given_dist_keys - required_dist_keys
            if miss_dist_keys:
                raise ValueError(
                    err_msg + f'Missing `{dist_key}` keys: '
                    f'{", ".join([f"{i}" for i in miss_dist_keys])}'
                )
            if bad_dist_keys:
                raise ValueError(
                    err_msg + f'Unknown `{dist_key}` keys: '
                    f'{", ".join([f"{i}" for i in bad_dist_keys])}'
                )

            # Match number of distributions to number of bins:
            for dist_param in required_dist_keys:  # i.e. "alpha" and "beta" for beta dist

                dist_param_val = dist[dist_param]

                if isinstance(dist_param_val, np.ndarray):
                    dist_param_val = dist_param_val.tolist()

                if not isinstance(dist_param_val, list):
                    dist_param_val = [dist_param_val]

                if len(dist_param_val) == 1:
                    dist_param_val = dist_param_val * num_bins

                elif len(dist_param_val) != num_bins:
                    raise ValueError(
                        err_msg + f'Distribution `{dist_key}` key "{dist_param}" must '
                        f'have length one, or length equal to the number of size '
                        f'distribution bins, which is {num_bins}, but in fact has '
                        f'length {len(dist_param_val)}.'
                    )
                dist[dist_param] = dist_param_val

            all_dists.update({dist_key: dist})

        # ODF:
        ODF_weights = {}
        axis_ODF_weights = {}
        ODF = phase_stats.get('ODF')
        axis_ODF = phase_stats.get('axis_ODF')

        if ODF or (phase_idx == 0 and orientations is not None):
            if not ODF:
                ODF = {}
            given_ODF_keys = set(ODF.keys())
            miss_ODF_keys = REQUIRED_PHASE_ODF_KEYS - given_ODF_keys
            bad_ODF_keys = given_ODF_keys - ALLOWED_PHASE_ODF_KEYS
            if miss_ODF_keys:
                raise ValueError(
                    err_msg + f'Missing `ODF` keys: '
                    f'{", ".join([f"{i}" for i in miss_ODF_keys])}'
                )
            if bad_ODF_keys:
                raise ValueError(
                    err_msg + f'Unknown `ODF` keys: '
                    f'{", ".join([f"{i}" for i in bad_ODF_keys])}'
                )

            ODF_presets = ODF.get('presets')

            if phase_idx == 0 and orientations is not None:
                # ALlow importing orientations only for the first phase:

                if ODF_presets:
                    warnings.warn(err_msg + f'Using locally defined ODF presets; not '
                                  f'using `orientations` from a previous task.')

                elif 'orientations' in ODF:
                    warnings.warn(err_msg + f'Using locally defined `orientations`, not '
                                  f'those from a previous task!')

                else:
                    ODF['orientations'] = orientations

            if ODF_presets:

                if any([ODF.get(i) is not None for i in ALLOWED_PHASE_AXIS_ODF_KEYS]):
                    raise ValueError(
                        err_msg + f'Specify either `presets` or `orientations` (and '
                        f'`sigmas and `weights).'
                    )
                preset_eulers = []
                preset_weights = []
                preset_sigmas = []
                for ODF_preset_idx, ODF_preset in enumerate(ODF_presets):
                    if (
                        'name' not in ODF_preset or
                        ODF_preset['name'].lower() not in ODF_CUBIC_PRESETS
                    ):
                        raise ValueError(
                            err_msg + f'Specify `name` for ODF preset index '
                            f'{ODF_preset_idx}; one of: '
                            f'{", ".join([f"{i}" for i in ODF_CUBIC_PRESETS.keys()])}'
                        )
                    preset_eulers.append(ODF_CUBIC_PRESETS[ODF_preset['name'].lower()])
                    preset_weights.append(ODF_preset.get('weight', DEFAULT_ODF_WEIGHT))
                    preset_sigmas.append(ODF_preset.get('sigma', DEFAULT_ODF_SIGMA))

                ODF['sigmas'] = preset_sigmas
                ODF['weights'] = preset_weights
                ODF['orientations'] = process_dream3D_euler_angles(
                    np.array(preset_eulers),
                    degrees=True,
                )

            oris = validate_orientations(ODF['orientations'])  # now as quaternions

            # Convert unit-cell alignment to x//a, as used by Dream.3D:
            if phase_i_CS == 'hexagonal':
                if oris['unit_cell_alignment'].get('y') == 'b':
                    hex_transform_quat = axang2quat(
                        oris['P'] * np.array([0, 0, 1]),
                        np.pi/6
                    )
                    for ori_idx, ori_i in enumerate(oris['quaternions']):
                        oris['quaternions'][ori_idx] = multiply_quaternions(
                            q1=hex_transform_quat,
                            q2=ori_i,
                            P=oris['P'],
                        )
                elif oris['unit_cell_alignment'].get('x') != 'a':
                    msg = (f'Cannot convert from the following specified unit cell '
                           f'alignment to Dream3D-compatible unit cell alignment (x//a): '
                           f'{oris["unit_cell_alignment"]}')
                    NotImplementedError(msg)

            num_oris = oris['quaternions'].shape[0]

            # Add defaults:
            if 'weights' not in ODF:
                ODF['weights'] = DEFAULT_ODF_WEIGHT
            if 'sigmas' not in ODF:
                ODF['sigmas'] = DEFAULT_ODF_SIGMA

            for i in ('weights', 'sigmas'):

                val = ODF[i]

                if isinstance(val, np.ndarray):
                    dist_param_val = val.tolist()

                if not isinstance(val, list):
                    val = [val]

                if len(val) == 1:
                    val = val * num_oris

                elif len(val) != num_oris:
                    raise ValueError(
                        err_msg + f'ODF key "{i}" must have length one, or length equal '
                        f'to the number of ODF orientations, which is {num_oris}, but in '
                        f'fact has length {len(val)}.'
                    )
                ODF[i] = val

            # Convert to Euler angles for Dream3D:
            oris_euler = quat2euler(oris['quaternions'], degrees=False, P=oris['P'])

            ODF_weights = {
                'Euler 1': oris_euler[:, 0].tolist(),
                'Euler 2': oris_euler[:, 1].tolist(),
                'Euler 3': oris_euler[:, 2].tolist(),
                'Sigma': ODF['sigmas'],
                'Weight': ODF['weights'],
            }

        if axis_ODF:
            given_axis_ODF_keys = set(axis_ODF.keys())
            miss_axis_ODF_keys = REQUIRED_PHASE_AXIS_ODF_KEYS - given_axis_ODF_keys
            bad_axis_ODF_keys = given_axis_ODF_keys - ALLOWED_PHASE_AXIS_ODF_KEYS
            if miss_axis_ODF_keys:
                raise ValueError(
                    err_msg + f'Missing `axis_ODF` keys: '
                    f'{", ".join([f"{i}" for i in miss_axis_ODF_keys])}'
                )
            if bad_axis_ODF_keys:
                raise ValueError(
                    err_msg + f'Unknown `axis_ODF` keys: '
                    f'{", ".join([f"{i}" for i in bad_axis_ODF_keys])}'
                )

            axis_oris = validate_orientations(
                axis_ODF['orientations'])  # now as quaternions

            # Convert unit-cell alignment to x//a, as used by Dream.3D:
            if phase_i_CS == 'hexagonal':
                if axis_oris['unit_cell_alignment'].get('y') == 'b':
                    hex_transform_quat = axang2quat(
                        axis_oris['P'] * np.array([0, 0, 1]),
                        np.pi/6
                    )
                    for ori_idx, ori_i in enumerate(axis_oris['quaternions']):
                        axis_oris['quaternions'][ori_idx] = multiply_quaternions(
                            q1=hex_transform_quat,
                            q2=ori_i,
                            P=axis_oris['P'],
                        )
                elif axis_oris['unit_cell_alignment'].get('x') != 'a':
                    msg = (f'Cannot convert from the following specified unit cell '
                           f'alignment to Dream3D-compatible unit cell alignment (x//a): '
                           f'{axis_oris["unit_cell_alignment"]}')
                    NotImplementedError(msg)

            num_oris = axis_oris['quaternions'].shape[0]

            # Add defaults:
            if 'weights' not in axis_ODF:
                axis_ODF['weights'] = DEFAULT_AXIS_ODF_WEIGHT
            if 'sigmas' not in axis_ODF:
                axis_ODF['sigmas'] = DEFAULT_AXIS_ODF_SIGMA

            for i in ('weights', 'sigmas'):

                val = axis_ODF[i]

                if isinstance(val, np.ndarray):
                    dist_param_val = val.tolist()

                if not isinstance(val, list):
                    val = [val]

                if len(val) == 1:
                    val = val * num_oris

                elif len(val) != num_oris:
                    raise ValueError(
                        err_msg +
                        f'axis_ODF key "{i}" must have length one, or length equal '
                        f'to the number of axis_ODF orientations, which is {num_oris}, but in '
                        f'fact has length {len(val)}.'
                    )
                axis_ODF[i] = val

            # Convert to Euler angles for Dream3D:
            axis_oris_euler = quat2euler(
                axis_oris['quaternions'], degrees=False, P=axis_oris['P'])

            axis_ODF_weights = {
                'Euler 1': axis_oris_euler[:, 0].tolist(),
                'Euler 2': axis_oris_euler[:, 1].tolist(),
                'Euler 3': axis_oris_euler[:, 2].tolist(),
                'Sigma': axis_ODF['sigmas'],
                'Weight': axis_ODF['weights'],
            }

        stats_JSON_i = {
            "AxisODF-Weights": axis_ODF_weights,
            "Bin Count": num_bins,
            "BinNumber": bins.tolist(),
            "BoundaryArea": 0,
            "Crystal Symmetry": ALLOWED_CRYSTAL_STRUCTURES[phase_i_CS],
            "FeatureSize Distribution": {
                "Average": log_mean,
                "Standard Deviation": sigma,
            },
            'Feature_Diameter_Info': feat_diam_info,
            'MDF-Weights': {},
            'ODF-Weights': ODF_weights,
            'Name': phase_stats['name'],
            'PhaseFraction': phase_stats['volume_fraction'],
            'PhaseType': phase_stats['type'].title(),
        }

        # Generate dists from `preset_statistics_model`:
        if preset:

            if 'omega3' not in all_dists:
                omega3_dist = generate_omega3_dist_from_preset(num_bins)
                all_dists.update({'omega3': omega3_dist})

            if 'c/a' not in all_dists:
                c_a_aspect_ratio = preset['A_axis_length'] / preset['C_axis_length']
                c_a_dist = generate_shape_dist_from_preset(
                    num_bins,
                    c_a_aspect_ratio,
                    preset_type,
                )
                all_dists.update({'c/a': c_a_dist})

            if 'b/a' not in all_dists:
                b_a_aspect_ratio = preset['A_axis_length'] / preset['B_axis_length']
                b_a_dist = generate_shape_dist_from_preset(
                    num_bins,
                    b_a_aspect_ratio,
                    preset_type,
                )
                all_dists.update({'b/a': b_a_dist})

            if phase_type == 'primary':
                if 'neighbours' not in all_dists:
                    neigh_dist = generate_neighbour_dist_from_preset(
                        num_bins,
                        preset_type,
                    )
                    all_dists.update({'neighbours': neigh_dist})

        # Coerce distributions into format expected in the JSON:
        for dist_key, dist in all_dists.items():
            dist_info = DISTRIBUTIONS_MAP[dist_key]
            stats_JSON_i.update({
                dist_info['label']: {
                    **{DISTRIBUTIONS_KEY_LABELS[k]: v for k, v in dist.items()},
                    'Distribution Type': DISTRIBUTIONS_TYPE_LABELS[dist_info['type']],
                }
            })

        if phase_stats['type'] == 'precipitate':
            stats_JSON_i.update({
                'Precipitate Boundary Fraction': phase_stats['number_fraction_on_boundary'],
                'Radial Distribution Function': {
                    'Bin Count': RDF['num_bins'],
                    'BoxDims': np.array(RDF['box_size']).tolist(),
                    'BoxRes': [  # TODO: how is this calculated?
                        0.1,
                        0.1,
                        0.1,
                    ],
                    'Max': RDF['max_distance'],
                    'Min': RDF['min_distance'],
                }
            })

        stats_JSON.append(stats_JSON_i)

    if not np.isclose(vol_frac_sum, 1.0):
        raise ValueError(f'Sum of `volume_fraction`s over all phases must sum to 1.0, '
                         f'but in fact sum to: {vol_frac_sum}')

    stats_data_array = {
        'Name': 'Statistics',
        'Phase Count': len(stats_JSON) + 1,  # Don't know why this needs to be +1
    }
    for idx, i in enumerate(stats_JSON, start=1):
        stats_data_array.update({str(idx): i})

    if precipitates:
        precip_inp_file = str(Path(path).parent.joinpath('precipitates.txt'))
    else:
        precip_inp_file = ""

    pipeline = {
        "0": {
            "CellEnsembleAttributeMatrixName": "CellEnsembleData",
            "CrystalStructuresArrayName": "CrystalStructures",
            "Filter_Enabled": True,
            "Filter_Human_Label": "StatsGenerator",
            "Filter_Name": "StatsGeneratorFilter",
            "Filter_Uuid": "{f642e217-4722-5dd8-9df9-cee71e7b26ba}",
            "PhaseNamesArrayName": "PhaseName",
            "PhaseTypesArrayName": "PhaseTypes",
            "StatsDataArray": stats_data_array,
            "StatsDataArrayName": "Statistics",
            "StatsGeneratorDataContainerName": "StatsGeneratorDataContainer"
        },
        "1": {
            # TODO: fix this
            "BoxDimensions": "X Range: 0 to 2 (Delta: 2)\nY Range: 0 to 256 (Delta: 256)\nZ Range: 0 to 256 (Delta: 256)",
            "CellAttributeMatrixName": "CellData",
            "DataContainerName": "SyntheticVolumeDataContainer",
            "Dimensions": {
                "x": grid_size[0],
                "y": grid_size[1],
                "z": grid_size[2]
            },
            "EnsembleAttributeMatrixName": "CellEnsembleData",
            "EstimateNumberOfFeatures": 0,
            "EstimatedPrimaryFeatures": "",
            "FilterVersion": "6.5.141",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Initialize Synthetic Volume",
            "Filter_Name": "InitializeSyntheticVolume",
            "Filter_Uuid": "{c2ae366b-251f-5dbd-9d70-d790376c0c0d}",
            "InputPhaseTypesArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "PhaseTypes",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "InputStatsArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "Statistics",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "Origin": {
                "x": origin[0],
                "y": origin[1],
                "z": origin[2]
            },
            "Resolution": {
                "x": resolution[0],
                "y": resolution[1],
                "z": resolution[2]
            }
        },
        "2": {
            "FilterVersion": "6.5.141",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Establish Shape Types",
            "Filter_Name": "EstablishShapeTypes",
            "Filter_Uuid": "{4edbbd35-a96b-5ff1-984a-153d733e2abb}",
            "InputPhaseTypesArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "PhaseTypes",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "ShapeTypeData": [
                999,
                0,
                0
            ],
            "ShapeTypesArrayName": "ShapeTypes"
        },
        "3": {
            "CellPhasesArrayName": "Phases",
            "FeatureGeneration": 0,
            "FeatureIdsArrayName": "FeatureIds",
            "FeatureInputFile": "",
            "FeaturePhasesArrayName": "Phases",
            "FilterVersion": "6.5.141",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Pack Primary Phases",
            "Filter_Name": "PackPrimaryPhases",
            "Filter_Uuid": "{84305312-0d10-50ca-b89a-fda17a353cc9}",
            "InputPhaseNamesArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "PhaseName",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "InputPhaseTypesArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "PhaseTypes",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "InputShapeTypesArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "ShapeTypes",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "InputStatsArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "Statistics",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "MaskArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "NewAttributeMatrixPath": {
                "Attribute Matrix Name": "Synthetic Shape Parameters (Primary Phase)",
                "Data Array Name": "",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "NumFeaturesArrayName": "NumFeatures",
            "OutputCellAttributeMatrixPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "OutputCellEnsembleAttributeMatrixName": "CellEnsembleData",
            "OutputCellFeatureAttributeMatrixName": "Grain Data",
            "PeriodicBoundaries": int(periodic),
            "SaveGeometricDescriptions": 0,
            "SelectedAttributeMatrixPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "UseMask": 0
        },
        "4": {
            "BoundaryCellsArrayName": "BoundaryCells",
            "FeatureIdsArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "FeatureIds",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "FilterVersion": "6.5.141",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Find Boundary Cells (Image)",
            "Filter_Name": "FindBoundaryCells",
            "Filter_Uuid": "{8a1106d4-c67f-5e09-a02a-b2e9b99d031e}",
            "IgnoreFeatureZero": 1,
            "IncludeVolumeBoundary": 0
        },
        "5": {
            "BoundaryCellsArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "BoundaryCells",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "CellPhasesArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "Phases",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "FeatureGeneration": 1 if precipitates else 0, # bug? should be opposite?
            "FeatureIdsArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "FeatureIds",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "FeaturePhasesArrayPath": {
                "Attribute Matrix Name": "Grain Data",
                "Data Array Name": "Phases",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "FilterVersion": "6.5.141",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Insert Precipitate Phases",
            "Filter_Name": "InsertPrecipitatePhases",
            "Filter_Uuid": "{1e552e0c-53bb-5ae1-bd1c-c7a6590f9328}",
            "InputPhaseTypesArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "PhaseTypes",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "InputShapeTypesArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "ShapeTypes",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "InputStatsArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "Statistics",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "MaskArrayPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "MatchRDF": 0,
            "NewAttributeMatrixPath": {
                "Attribute Matrix Name": "Synthetic Shape Parameters (Precipitate)",
                "Data Array Name": "",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "NumFeaturesArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "NumFeatures",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "PeriodicBoundaries": int(periodic),
            "PrecipInputFile": precip_inp_file,
            "SaveGeometricDescriptions": 0,
            "SelectedAttributeMatrixPath": {
                "Attribute Matrix Name": "",
                "Data Array Name": "",
                "Data Container Name": ""
            },
            "UseMask": 0
        },
        "6": {
            "BoundaryCellsArrayName": "BoundaryCells",
            "CellFeatureAttributeMatrixPath": {
                "Attribute Matrix Name": "Grain Data",
                "Data Array Name": "",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "FeatureIdsArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "FeatureIds",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "FilterVersion": "6.5.141",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Find Feature Neighbors",
            "Filter_Name": "FindNeighbors",
            "Filter_Uuid": "{97cf66f8-7a9b-5ec2-83eb-f8c4c8a17bac}",
            "NeighborListArrayName": "NeighborList",
            "NumNeighborsArrayName": "NumNeighbors",
            "SharedSurfaceAreaListArrayName": "SharedSurfaceAreaList",
            "StoreBoundaryCells": 0,
            "StoreSurfaceFeatures": 1,
            "SurfaceFeaturesArrayName": "SurfaceFeatures"
        },
        "7": {
            "AvgQuatsArrayName": "AvgQuats",
            "CellEulerAnglesArrayName": "EulerAngles",
            "CrystalStructuresArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "CrystalStructures",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "FeatureEulerAnglesArrayName": "EulerAngles",
            "FeatureIdsArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "FeatureIds",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "FeaturePhasesArrayPath": {
                "Attribute Matrix Name": "Grain Data",
                "Data Array Name": "Phases",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "FilterVersion": "6.5.141",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Match Crystallography",
            "Filter_Name": "MatchCrystallography",
            "Filter_Uuid": "{7bfb6e4a-6075-56da-8006-b262d99dff30}",
            "InputStatsArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "Statistics",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "MaxIterations": 100000,
            "NeighborListArrayPath": {
                "Attribute Matrix Name": "Grain Data",
                "Data Array Name": "NeighborList",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "NumFeaturesArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "NumFeatures",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "PhaseTypesArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "PhaseTypes",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "SharedSurfaceAreaListArrayPath": {
                "Attribute Matrix Name": "Grain Data",
                "Data Array Name": "SharedSurfaceAreaList",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "SurfaceFeaturesArrayPath": {
                "Attribute Matrix Name": "Grain Data",
                "Data Array Name": "SurfaceFeatures",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "VolumesArrayName": "Volumes"
        },
        "8": {
            "CellEulerAnglesArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "EulerAngles",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "CellIPFColorsArrayName": "IPFColor",
            "CellPhasesArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "Phases",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "CrystalStructuresArrayPath": {
                "Attribute Matrix Name": "CellEnsembleData",
                "Data Array Name": "CrystalStructures",
                "Data Container Name": "StatsGeneratorDataContainer"
            },
            "FilterVersion": "6.5.141",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Generate IPF Colors",
            "Filter_Name": "GenerateIPFColors",
            "Filter_Uuid": "{a50e6532-8075-5de5-ab63-945feb0de7f7}",
            "GoodVoxelsArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "",
                "Data Container Name": "SyntheticVolumeDataContainer"
            },
            "ReferenceDir": {
                "x": 0,
                "y": 0,
                "z": 1
            },
            "UseGoodVoxels": 0
        },
        "9": {
            "FilterVersion": "1.2.675",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Write DREAM.3D Data File",
            "Filter_Name": "DataContainerWriter",
            "Filter_Uuid": "{3fcd4c43-9d75-5b86-aad4-4441bc914f37}",
            "OutputFile": f"{str(Path(path).parent.joinpath('pipeline.dream3d'))}",
            "WriteTimeSeries": 0,
            "WriteXdmfFile": 1
        },
        "PipelineBuilder": {
            "Name": "RVE from precipitate statistics",
            "Number_Filters": 10,
            "Version": 6
        }
    }

    with Path(path).open('w') as fh:
        json.dump(pipeline, fh, indent=4)
