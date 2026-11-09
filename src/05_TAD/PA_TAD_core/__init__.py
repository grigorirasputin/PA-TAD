"""
PA_TAD_core: shared layers, graph utilities, data pipeline, and evaluation helpers
for PA-TAD and PA-TAD-Lite.
"""

from .data_utils import (
    TrajectoryDataset,
    build_graph_and_mapping,
    collate_fn,
    extract_linestring_geometry_features,
    extract_spatial_subtype,
)
from .layers import BinaryFocalLoss, MultiTaskLossWrapper, PA_TAD_Lite
from .tools import EarlyStopping, compute_trip_scores, evaluate_model, set_all_seeds, setup_logger

__all__ = [
    "TrajectoryDataset",
    "build_graph_and_mapping",
    "collate_fn",
    "extract_linestring_geometry_features",
    "extract_spatial_subtype",
    "PA_TAD_Lite",
    "BinaryFocalLoss",
    "MultiTaskLossWrapper",
    "EarlyStopping",
    "compute_trip_scores",
    "evaluate_model",
    "set_all_seeds",
    "setup_logger",
]
