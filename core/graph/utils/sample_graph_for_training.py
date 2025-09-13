#!/usr/bin/env python3
"""
Sample a manageable subgraph from the massive graph for GraphSAGE training.
Uses intelligent sampling to preserve graph properties.
"""

import os
import json
import logging
import numpy as np
from collections import defaultdict, deque
from tqdm import tqdm
from arango import ArangoClient
import click

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GraphSampler:
    """Sample subgraphs from massive graphs while preserving key properties."""
    
    def __init__(self, db_name: str = 'academy_store'):
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db(
            db_name,
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
    
    def sample_by_random_walk(self, num_nodes: int = 100000, walk_length: int = 10, 
                            num_walks: int = 5) -> set:
        """Sample nodes using random walks to preserve local structure."""
        logger.info(f"Sampling {num_nodes} nodes using random walks...")
        
        sampled_nodes = set()
        
        # Get seed nodes
        query = """
        FOR p IN arxiv_papers
            SORT RAND()
            LIMIT @num_seeds
            RETURN p._key
        """
        
        num_seeds = min(1000, num_nodes // 100)
        seed_nodes = list(self.db.aql.execute(query, bind_vars={'num_seeds': num_seeds}))
        
        # Perform random walks from each seed
        for seed in tqdm(seed_nodes, desc="Random walks"):
            for _ in range(num_walks):
                current = seed
                walk = [current]
                
                for _ in range(walk_length):
                    # Get random neighbor
                    neighbor_query = """
                    FOR v IN 1..1 ANY CONCAT('arxiv_papers/', @node) 
                        coauthorship, same_field
                        SORT RAND()
                        LIMIT 1
                        RETURN PARSE_IDENTIFIER(v._id).key
                    """
                    
                    neighbors = list(self.db.aql.execute(
                        neighbor_query, 
                        bind_vars={'node': current}
                    ))
                    
                    if neighbors:
                        current = neighbors[0]
                        walk.append(current)
                    else:
                        break
                
                sampled_nodes.update(walk)
                
                if len(sampled_nodes) >= num_nodes:
                    break
            
            if len(sampled_nodes) >= num_nodes:
                break
        
        sampled_nodes = list(sampled_nodes)[:num_nodes]
        logger.info(f"Sampled {len(sampled_nodes)} nodes")
        
        return set(sampled_nodes)
    
    def sample_by_bfs(self, num_nodes: int = 100000, num_seeds: int = 100) -> set:
        """Sample a connected component using BFS from seed nodes."""
        logger.info(f"Sampling {num_nodes} nodes using BFS...")
        
        # Get high-degree seed nodes for better connectivity
        query = """
        FOR p IN arxiv_papers
            LET degree = LENGTH(
                FOR e IN coauthorship
                    FILTER e._from == CONCAT('arxiv_papers/', p._key) 
                       OR e._to == CONCAT('arxiv_papers/', p._key)
                    LIMIT 100
                    RETURN 1
            )
            FILTER degree > 10
            SORT degree DESC
            LIMIT @num_seeds
            RETURN p._key
        """
        
        seed_nodes = list(self.db.aql.execute(query, bind_vars={'num_seeds': num_seeds}))
        
        if not seed_nodes:
            # Fallback to random seeds
            query = """
            FOR p IN arxiv_papers
                SORT RAND()
                LIMIT @num_seeds
                RETURN p._key
            """
            seed_nodes = list(self.db.aql.execute(query, bind_vars={'num_seeds': num_seeds}))
        
        sampled_nodes = set()
        queue = deque(seed_nodes)
        
        while queue and len(sampled_nodes) < num_nodes:
            current = queue.popleft()
            
            if current in sampled_nodes:
                continue
            
            sampled_nodes.add(current)
            
            # Get neighbors
            neighbor_query = """
            FOR v IN 1..1 ANY CONCAT('arxiv_papers/', @node) 
                coauthorship, same_field
                LIMIT 50
                RETURN PARSE_IDENTIFIER(v._id).key
            """
            
            neighbors = list(self.db.aql.execute(
                neighbor_query,
                bind_vars={'node': current}
            ))
            
            for neighbor in neighbors:
                if neighbor not in sampled_nodes:
                    queue.append(neighbor)
            
            if len(sampled_nodes) % 10000 == 0:
                logger.info(f"  Sampled {len(sampled_nodes)} nodes...")
        
        sampled_nodes = list(sampled_nodes)[:num_nodes]
        logger.info(f"Sampled {len(sampled_nodes)} nodes via BFS")
        
        return set(sampled_nodes)
    
    def sample_by_categories(self, num_nodes: int = 100000, 
                            categories: list = None) -> set:
        """Sample nodes from specific categories."""
        logger.info(f"Sampling {num_nodes} nodes by categories...")
        
        if not categories:
            # Get top categories
            query = """
            FOR p IN arxiv_papers
                FILTER p.categories != null
                FOR cat IN p.categories
                    COLLECT category = cat WITH COUNT INTO count
                    SORT count DESC
                    LIMIT 10
                    RETURN category
            """
            categories = list(self.db.aql.execute(query))
            logger.info(f"Using top categories: {categories[:5]}")
        
        nodes_per_category = num_nodes // len(categories)
        sampled_nodes = set()
        
        for category in categories:
            query = """
            FOR p IN arxiv_papers
                FILTER @category IN p.categories
                SORT RAND()
                LIMIT @limit
                RETURN p._key
            """
            
            nodes = list(self.db.aql.execute(
                query,
                bind_vars={'category': category, 'limit': nodes_per_category}
            ))
            
            sampled_nodes.update(nodes)
            logger.info(f"  {category}: {len(nodes)} nodes")
        
        sampled_nodes = list(sampled_nodes)[:num_nodes]
        logger.info(f"Sampled {len(sampled_nodes)} nodes from {len(categories)} categories")
        
        return set(sampled_nodes)
    
    def extract_subgraph(self, node_set: set, output_path: str):
        """Extract edges for sampled nodes and save as graph."""
        logger.info(f"Extracting subgraph for {len(node_set)} nodes...")
        
        node_list = list(node_set)
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        # Get all edges between sampled nodes
        edge_collections = ['coauthorship', 'same_field', 'temporal_proximity', 'citations']
        all_edges = []
        adjacency = defaultdict(list)
        
        for coll_name in edge_collections:
            if coll_name not in [c['name'] for c in self.db.collections()]:
                continue
            
            logger.info(f"  Extracting {coll_name} edges...")
            
            # Process in batches
            batch_size = 1000
            edge_count = 0
            
            for i in range(0, len(node_list), batch_size):
                batch = node_list[i:i+batch_size]
                
                query = f"""
                FOR e IN {coll_name}
                    LET from_key = PARSE_IDENTIFIER(e._from).key
                    LET to_key = PARSE_IDENTIFIER(e._to).key
                    FILTER from_key IN @nodes AND to_key IN @nodes
                    RETURN {{
                        from: from_key,
                        to: to_key,
                        weight: e.weight
                    }}
                """
                
                edges = list(self.db.aql.execute(query, bind_vars={'nodes': batch}))
                
                for edge in edges:
                    if edge['from'] in node_to_idx and edge['to'] in node_to_idx:
                        from_idx = node_to_idx[edge['from']]
                        to_idx = node_to_idx[edge['to']]
                        
                        adjacency[from_idx].append(to_idx)
                        adjacency[to_idx].append(from_idx)
                        
                        all_edges.append({
                            'from': from_idx,
                            'to': to_idx,
                            'type': coll_name,
                            'weight': edge.get('weight', 1.0)
                        })
                        edge_count += 1
            
            logger.info(f"    Found {edge_count} {coll_name} edges")
        
        # Get node metadata
        logger.info("Loading node metadata...")
        node_metadata = {}
        
        query = """
        FOR p IN arxiv_papers
            FILTER p._key IN @nodes
            RETURN {
                id: p._key,
                title: p.title,
                categories: p.categories,
                year: SUBSTRING(p.update_date, 0, 4)
            }
        """
        
        for node in self.db.aql.execute(query, bind_vars={'nodes': node_list}):
            node_metadata[node['id']] = node
        
        # Create graph data structure
        graph_data = {
            'num_nodes': len(node_list),
            'num_edges': len(all_edges),
            'node_ids': node_list,
            'node_to_idx': node_to_idx,
            'adjacency': dict(adjacency),
            'edges': all_edges[:1000000],  # Cap edges at 1M for memory
            'node_metadata': node_metadata,
            'sampling_method': 'subgraph'
        }
        
        # Save graph
        logger.info(f"Saving graph to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(graph_data, f)
        
        logger.info(f"Saved subgraph: {len(node_list):,} nodes, {len(all_edges):,} edges")
        
        return graph_data


@click.command()
@click.option('--method', type=click.Choice(['random_walk', 'bfs', 'category']), 
              default='bfs', help='Sampling method')
@click.option('--num-nodes', default=100000, help='Number of nodes to sample')
@click.option('--output', default='sampled_graph.json', help='Output path')
@click.option('--categories', multiple=True, help='Specific categories to sample')
def main(method, num_nodes, output, categories):
    """Sample a subgraph from the massive academic graph."""
    
    sampler = GraphSampler()
    
    if method == 'random_walk':
        node_set = sampler.sample_by_random_walk(num_nodes)
    elif method == 'bfs':
        node_set = sampler.sample_by_bfs(num_nodes)
    elif method == 'category':
        node_set = sampler.sample_by_categories(num_nodes, list(categories))
    
    # Extract and save subgraph
    sampler.extract_subgraph(node_set, output)
    
    logger.info(f"Sampling complete! Graph saved to {output}")
    logger.info("You can now train GraphSAGE with:")
    logger.info(f"  python train_graphsage.py --graph-path {output}")


if __name__ == '__main__':
    main()