"""Navigation module for pathfinding and world mapping."""

from .world_map import WorldMapManager, ZoneMap, POI
from .minimap_detector import MinimapPOIDetector, MinimapPOI
from .navigator import Navigator, NavigationPath
from .click_navigator import ClickNavigator, NavigationGoal, NavigationMode
from .map_geometry import GeometryResult, MinimapGeometryExtractor

# Vision-based navigation components
from .minimap_extractor import MinimapExtractor
from .minimap_processor import MinimapProcessor, MinimapGrid, CellType
from .local_map import LocalMap, Frontier
from .pose_estimator import PoseEstimator, Pose
from .frontier_navigator import FrontierNavigator, NavigationAction, NavigationState
from .nav_visualization import NavigationOverlay, draw_pose_arrow, draw_path_on_frame
from .static_map_localizer import StaticMapLocalizer, load_zone_static_map, LocalizationResult

# ECC-based advanced localization
from .image_preprocessing import OrientedFilterBank, OrientedMorphology, AdaptiveThresholdProcessor
from .minimap_edge_extractor import MinimapEdgeExtractor
from .ecc_localizer import ECCAligner, StaticMapMatcher
from .static_map_localizer_base import StaticMapLocalizerBase
from .ecc_static_localizer import ECCStaticMapLocalizer
from .ransac_static_localizer import RANSACStaticMapLocalizer

# SLAM system (visual odometry, no game coordinates)
from .minimap_slam import MinimapSLAM, OccupancyCell, POI as SLAM_POI, MapSignature, Level
from .slam_visualizer import SLAMVisualizer

__all__ = [
    'WorldMapManager',
    'ZoneMap',
    'POI',
    'MinimapPOIDetector',
    'MinimapPOI',
    'Navigator',
    'NavigationPath',
    'ClickNavigator',
    'NavigationGoal',
    'NavigationMode',
    'GeometryResult',
    'MinimapGeometryExtractor',
    # Vision-based navigation
    'MinimapExtractor',
    'MinimapProcessor',
    'MinimapGrid',
    'CellType',
    'LocalMap',
    'Frontier',
    'PoseEstimator',
    'Pose',
    'FrontierNavigator',
    'NavigationAction',
    'NavigationState',
    'NavigationOverlay',
    'draw_pose_arrow',
    'draw_path_on_frame',
    'StaticMapLocalizer',
    'load_zone_static_map',
    'LocalizationResult',
    # ECC-based advanced localization
    'OrientedFilterBank',
    'OrientedMorphology',
    'AdaptiveThresholdProcessor',
    'MinimapEdgeExtractor',
    'ECCAligner',
    'StaticMapMatcher',
    'StaticMapLocalizerBase',
    'ECCStaticMapLocalizer',
    'RANSACStaticMapLocalizer',
    # SLAM system
    'MinimapSLAM',
    'OccupancyCell',
    'SLAM_POI',
    'MapSignature',
    'Level',
    'SLAMVisualizer',
]
