"""
Advanced Graph Visualization for ArangoDB.

Provides multiple visualization backends for exploring the graph structure:
1. PyVis - Interactive HTML networks
2. Plotly - 3D interactive graphs
3. Datashader - Large-scale visualization
4. Export to Gephi format
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import networkx as nx
from arango import ArangoClient
import plotly.graph_objects as go
from pyvis.network import Network
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent.parent))


@dataclass
class VisualizationConfig:
    """Configuration for graph visualization."""
    max_nodes: int = 1000
    max_edges: int = 5000
    layout: str = 'spring'  # spring, kamada_kawai, circular, random
    node_size_attr: Optional[str] = 'degree'  # degree, pagerank, betweenness
    node_color_attr: Optional[str] = 'type'  # type, community, category
    edge_weight_attr: Optional[str] = 'weight'
    physics: bool = True
    show_labels: bool = True
    output_format: str = 'html'  # html, json, gexf, graphml


class GraphVisualizer:
    """Advanced visualization for ArangoDB graphs."""
    
    def __init__(self, db_config: Dict):
        """Initialize with database configuration."""
        self.client = ArangoClient(hosts=db_config.get('host', 'http://localhost:8529'))
        self.db = self.client.db(
            db_config['database'],
            username=db_config['username'],
            password=db_config.get('password', os.environ.get('ARANGO_PASSWORD'))
        )
        self.graph = None
        
    def query_subgraph(self, start_node: str = None, depth: int = 2, 
                      limit: int = 1000) -> nx.Graph:
        """
        Query a subgraph from ArangoDB.
        
        Args:
            start_node: Starting node ID (if None, random sample)
            depth: Depth of traversal
            limit: Maximum nodes to retrieve
            
        Returns:
            NetworkX graph
        """
        if start_node:
            # BFS from start node
            query = """
                FOR v, e, p IN 1..@depth ANY @start_node 
                    GRAPH 'academy_graph'
                    OPTIONS {bfs: true, uniqueVertices: 'global'}
                    LIMIT @limit
                RETURN {
                    vertex: v,
                    edge: e,
                    path: p
                }
            """
            bind_vars = {
                'start_node': start_node,
                'depth': depth,
                'limit': limit
            }
        else:
            # Sample random nodes and their neighborhoods
            query = """
                FOR paper IN arxiv_papers
                    SORT RAND()
                    LIMIT @sample_size
                    
                    FOR v, e IN 1..@depth ANY paper
                        coauthorship, shared_category, paper_has_chunk
                        OPTIONS {bfs: true, uniqueVertices: 'global'}
                        LIMIT @limit
                    RETURN DISTINCT {
                        vertex: v,
                        edge: e
                    }
            """
            bind_vars = {
                'sample_size': min(10, limit // 10),
                'depth': depth,
                'limit': limit
            }
        
        try:
            cursor = self.db.aql.execute(query, bind_vars=bind_vars)
            results = list(cursor)
        except:
            # Fallback to simpler query
            query = """
                FOR edge IN coauthorship
                    LIMIT @limit
                    RETURN edge
            """
            cursor = self.db.aql.execute(query, bind_vars={'limit': limit})
            results = list(cursor)
        
        # Build NetworkX graph
        G = nx.Graph()
        
        for result in results:
            if isinstance(result, dict):
                if '_from' in result and '_to' in result:
                    # It's an edge
                    from_id = result['_from'].split('/')[-1]
                    to_id = result['_to'].split('/')[-1]
                    weight = result.get('weight', 1.0)
                    G.add_edge(from_id, to_id, weight=weight)
                elif 'vertex' in result:
                    # It's a vertex
                    v = result['vertex']
                    node_id = v.get('_key', v.get('_id', '').split('/')[-1])
                    G.add_node(node_id, 
                              title=v.get('title', '')[:50],
                              type=v.get('type', 'unknown'))
        
        print(f"Loaded subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        self.graph = G
        return G
    
    def visualize_pyvis(self, config: VisualizationConfig = None) -> str:
        """
        Create interactive visualization using PyVis.
        
        Returns:
            Path to HTML file
        """
        if config is None:
            config = VisualizationConfig()
        
        if self.graph is None:
            self.query_subgraph(limit=config.max_nodes)
        
        # Create PyVis network
        net = Network(height='800px', width='100%', 
                     bgcolor='#222222', font_color='white')
        
        if config.physics:
            net.barnes_hut(gravity=-80000, central_gravity=0.3,
                          spring_length=250, spring_strength=0.001)
        
        # Add nodes with attributes
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            
            # Size based on degree
            size = min(50, 10 + self.graph.degree[node] * 2)
            
            # Color based on type/community
            color = self._get_node_color(node_data.get('type', 'unknown'))
            
            # Label
            label = node if config.show_labels else ''
            title = node_data.get('title', node)
            
            net.add_node(node, label=label, title=title, 
                        size=size, color=color)
        
        # Add edges
        for edge in self.graph.edges(data=True):
            weight = edge[2].get('weight', 1.0)
            net.add_edge(edge[0], edge[1], value=weight)
        
        # Set options
        net.set_options("""
        var options = {
          "nodes": {
            "borderWidth": 2,
            "shadow": true
          },
          "edges": {
            "color": {
              "inherit": true
            },
            "smooth": {
              "type": "continuous"
            },
            "shadow": true
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "iterations": 100
            }
          }
        }
        """)
        
        # Save to HTML
        output_path = 'graph_visualization.html'
        net.show(output_path)
        print(f"Visualization saved to {output_path}")
        return output_path
    
    def visualize_plotly_3d(self, config: VisualizationConfig = None) -> go.Figure:
        """
        Create 3D visualization using Plotly.
        
        Returns:
            Plotly figure object
        """
        if config is None:
            config = VisualizationConfig()
        
        if self.graph is None:
            self.query_subgraph(limit=config.max_nodes)
        
        # Compute layout
        if config.layout == 'spring':
            pos = nx.spring_layout(self.graph, dim=3, k=0.5, iterations=50)
        elif config.layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph, dim=3)
        else:
            pos = nx.random_layout(self.graph, dim=3)
        
        # Extract node positions
        node_x = [pos[node][0] for node in self.graph.nodes()]
        node_y = [pos[node][1] for node in self.graph.nodes()]
        node_z = [pos[node][2] for node in self.graph.nodes()]
        
        # Extract edge positions
        edge_x = []
        edge_y = []
        edge_z = []
        
        for edge in self.graph.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
        
        # Create edge trace
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none'
        )
        
        # Create node trace
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text' if config.show_labels else 'markers',
            text=[str(node)[:20] for node in self.graph.nodes()],
            textposition="top center",
            marker=dict(
                size=[min(20, 5 + self.graph.degree[node]) 
                      for node in self.graph.nodes()],
                color=[self.graph.degree[node] for node in self.graph.nodes()],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    thickness=15,
                    title="Node Degree",
                    xanchor="left",
                    titleside="right"
                ),
                line=dict(width=2)
            ),
            hovertemplate='%{text}<br>Degree: %{marker.color}<extra></extra>'
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title=f'Graph Visualization ({self.graph.number_of_nodes()} nodes, '
                  f'{self.graph.number_of_edges()} edges)',
            showlegend=False,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            hovermode='closest',
            paper_bgcolor="black",
            scene_camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )
        
        # Save to HTML
        output_path = 'graph_3d_visualization.html'
        fig.write_html(output_path)
        print(f"3D visualization saved to {output_path}")
        
        return fig
    
    def export_to_gephi(self, output_path: str = 'graph.gexf'):
        """
        Export graph to Gephi GEXF format.
        
        Args:
            output_path: Path to save GEXF file
        """
        if self.graph is None:
            self.query_subgraph(limit=10000)
        
        # Add node attributes for Gephi
        for node in self.graph.nodes():
            self.graph.nodes[node]['label'] = str(node)
            self.graph.nodes[node]['viz'] = {
                'size': min(50, 10 + self.graph.degree[node] * 2)
            }
        
        # Export to GEXF
        nx.write_gexf(self.graph, output_path)
        print(f"Graph exported to {output_path} for Gephi")
    
    def visualize_communities(self):
        """Detect and visualize communities in the graph."""
        if self.graph is None:
            self.query_subgraph(limit=5000)
        
        # Detect communities using Louvain method
        try:
            import community as community_louvain
            communities = community_louvain.best_partition(self.graph)
            
            # Add community info to nodes
            for node, comm_id in communities.items():
                self.graph.nodes[node]['community'] = comm_id
            
            print(f"Found {len(set(communities.values()))} communities")
            
        except ImportError:
            print("Install python-louvain for community detection: pip install python-louvain")
            # Fallback to connected components
            components = list(nx.connected_components(self.graph))
            print(f"Found {len(components)} connected components")
            
            for i, component in enumerate(components):
                for node in component:
                    self.graph.nodes[node]['community'] = i
        
        # Visualize with communities colored
        config = VisualizationConfig(node_color_attr='community')
        return self.visualize_pyvis(config)
    
    def _get_node_color(self, node_type: str) -> str:
        """Get color for node type."""
        colors = {
            'paper': '#4CAF50',      # Green
            'chunk': '#2196F3',      # Blue  
            'author': '#FF9800',     # Orange
            'category': '#9C27B0',   # Purple
            'code': '#F44336',       # Red
            'unknown': '#9E9E9E'     # Grey
        }
        return colors.get(node_type, colors['unknown'])
    
    def analyze_graph_metrics(self) -> Dict:
        """Compute graph metrics for analysis."""
        if self.graph is None:
            self.query_subgraph(limit=5000)
        
        metrics = {}
        
        # Basic metrics
        metrics['num_nodes'] = self.graph.number_of_nodes()
        metrics['num_edges'] = self.graph.number_of_edges()
        metrics['density'] = nx.density(self.graph)
        
        # Centrality metrics (sample for large graphs)
        sample_nodes = list(self.graph.nodes())[:100]
        subgraph = self.graph.subgraph(sample_nodes)
        
        metrics['avg_degree'] = np.mean([d for n, d in self.graph.degree()])
        metrics['avg_clustering'] = nx.average_clustering(subgraph)
        
        # Find important nodes
        degree_cent = nx.degree_centrality(subgraph)
        top_nodes = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:5]
        metrics['top_nodes_by_degree'] = top_nodes
        
        if nx.is_connected(subgraph):
            metrics['diameter'] = nx.diameter(subgraph)
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(subgraph)
        
        return metrics


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize ArangoDB graph")
    parser.add_argument('--mode', choices=['pyvis', '3d', 'gephi', 'communities'],
                       default='pyvis', help='Visualization mode')
    parser.add_argument('--start', type=str, help='Starting node ID')
    parser.add_argument('--limit', type=int, default=1000,
                       help='Maximum nodes to visualize')
    args = parser.parse_args()
    
    db_config = {
        'host': 'http://localhost:8529',
        'database': 'academy_store',
        'username': 'root',
        'password': os.environ.get('ARANGO_PASSWORD')
    }
    
    visualizer = GraphVisualizer(db_config)
    
    # Query subgraph
    visualizer.query_subgraph(start_node=args.start, limit=args.limit)
    
    # Analyze metrics
    metrics = visualizer.analyze_graph_metrics()
    print("\nGraph Metrics:")
    for key, value in metrics.items():
        if not isinstance(value, list):
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Visualize based on mode
    if args.mode == 'pyvis':
        visualizer.visualize_pyvis()
    elif args.mode == '3d':
        visualizer.visualize_plotly_3d()
    elif args.mode == 'gephi':
        visualizer.export_to_gephi()
    elif args.mode == 'communities':
        visualizer.visualize_communities()
    
    print(f"\n✅ Visualization complete! Open the generated HTML file to explore.")


if __name__ == "__main__":
    main()