'`matflow_dream3d.main.py`'

import json
from pathlib import Path
from textwrap import dedent

import h5py
import numpy as np

from matflow_dream3d import input_mapper, output_mapper
from matflow_dream3d.utilities import quat2euler

@output_mapper(
    output_name='volume_element',
    task='generate_volume_element',
    method='from_statistics',
)
@output_mapper(
    output_name='volume_element',
    task='segment_grains',
    method='burn',
)
def parse_dream_3D_volume_element(path):
    
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
        'orientations': {
            'type': 'euler',
            'euler_degrees': False,
            'euler_angles': eulers,
            'unit_cell_alignment': {'x': 'a'},
        },
    }
    return vol_elem

@input_mapper(
    input_file='orientation_data.txt',
    task='segment_grains',
    method='burn',
)
def write_segment_grains_orientations_file(path, volume_element_response):
        
    increment = -1

    phase = volume_element_response['field_data']['phase']['data']    
    ori_data = volume_element_response['field_data']['O']['data']['quaternions']
    
    oris_flat = ori_data[increment].reshape(-1, 4)    
    eulers = quat2euler(oris_flat)

    col_names = [
        'Phase',
        'Euler1',
        'Euler2',
        'Euler3',
    ]

    all_dat = np.hstack([phase.reshape(-1, 1), eulers])
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
def write_segment_grains_pipeline(path, volume_element):
    
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
                "Euler1",
                "Euler2",
                "Euler3"
            ],
            "Wizard_DataTypes": [
                "int32_t",
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
                    "Data Array Name": "Euler1",
                    "Data Container Name": "DataContainer"
                },
                {
                    "Attribute Matrix Name": "CellData",
                    "Data Array Name": "Euler2",
                    "Data Container Name": "DataContainer"
                },
                {
                    "Attribute Matrix Name": "CellData",
                    "Data Array Name": "Euler3",
                    "Data Container Name": "DataContainer"
                }
            ],
            "StackedDataArrayName": "Eulers"
        },
        "4": {
            "FilterVersion": "6.5.141",
            "Filter_Enabled": True,
            "Filter_Human_Label": "Convert Orientation Representation",
            "Filter_Name": "ConvertOrientations",
            "Filter_Uuid": "{e5629880-98c4-5656-82b8-c9fe2b9744de}",
            "InputOrientationArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "Eulers",
                "Data Container Name": "DataContainer"
            },
            "InputType": 0,
            "OutputOrientationArrayName": "quats",
            "OutputType": 2
        },
        "5": {
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
        "6": {
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
            "MisorientationTolerance": 7,
            "QuatsArrayPath": {
                "Attribute Matrix Name": "CellData",
                "Data Array Name": "quats",
                "Data Container Name": "DataContainer"
            },
            "UseGoodVoxels": 0
        },
        "7": {
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
            "Name": "test_pipeline_2",
            "Number_Filters": 8,
            "Version": 6
        }
    }
    with Path(path).open('w') as fh:
        json.dump(pipeline, fh, indent=4)


@input_mapper(
    input_file='pipeline.json',
    task='generate_volume_element',
    method='from_statistics',
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
