"""
GraphSAGE Tool - Theory-Practice Bridge Discovery.

Discovers connections between theoretical papers and practical implementations
using GraphSAGE's inductive graph learning capabilities.
"""

from .pipelines.graphsage_pipeline import GraphSAGEPipeline, PipelineConfig
from .bridge_discovery.theory_practice_finder import TheoryPracticeFinder, BridgeDiscoveryConfig
from .utils.neighborhood_sampler import NeighborhoodSampler, SamplingConfig

__all__ = [
    'GraphSAGEPipeline',
    'PipelineConfig',
    'TheoryPracticeFinder',
    'BridgeDiscoveryConfig',
    'NeighborhoodSampler',
    'SamplingConfig'
]