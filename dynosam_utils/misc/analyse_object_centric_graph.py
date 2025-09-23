import networkx as nx
import matplotlib.pyplot as plt

#sudo apt install graphviz-dev && pip install pygraphviz

path = "/root/results/misc/isam_graph_19_object_centric.dot"
n_partitions = 4

def draw_graph_from_dot(dot_filepath):
    """Reads a .dot file, draws the graph with node labels from metadata.

    Args:
        dot_filepath: Path to the .dot file.
    """
    try:
        # Read the graph from the .dot file
        graph = nx.drawing.nx_agraph.read_dot(dot_filepath)  # Handles graphviz attributes
        # graph = nx.Graph(nx.nx_pydot.read_dot(dot_filepath))

        # Extract labels from node metadata (if available)
        labels = {}
        for node, data in graph.nodes(data=True):
            # Check if 'label' key exists in the node's data
            if 'label' in data:
                labels[node] = data['label']
            else:
                # Fallback: use node name if no label in metadata
                labels[node] = str(node)  # Convert node to string if needed

        # Draw the graph
        pos = nx.drawing.nx_agraph.graphviz_layout(graph) # Use graphviz for better layout if available
        # nx.draw(graph, pos, with_labels=True, labels=labels, node_size=700, node_color="skyblue", font_size=10, font_color="black") # Adjust aesthetics as needed

        # plt.title("Graph from .dot file")
        # plt.show()
        import metis
        result, partition = metis.part_graph(graph, n_partitions, recursive=True)
        print(partition)
        # print(result)
        print(len(graph.nodes()))
        print(len(partition))

        cmap = plt.cm.get_cmap('viridis', max(partition) + 1)  # Color map
        # pos = nx.spring_layout(graph)  # You can try other layouts too
        nx.draw(graph, pos, node_color=[cmap(i) for i in partition], with_labels=True, labels=labels, node_size=700, font_size=10)
        # # plt.title(title)
        plt.show()

    except FileNotFoundError:
        print(f"Error: .dot file not found at {dot_filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")


def read_dot_and_project(dot_filepath):
    """Reads a .dot file, projects the bipartite graph onto the 'vars' set.

    Args:
        dot_filepath: Path to the .dot file.

    Returns:
        A NetworkX graph with only 'vars' as nodes, or None on error.
    """
    try:
        graph = nx.drawing.nx_agraph.read_dot(dot_filepath)

        vars_nodes = {}  # Store vars and their data
        factors = set()
        var_factor_edges = []



        for node, data in graph.nodes(data=True):
            if "var" in node:
                # vars_nodes.add(node)
                vars_nodes[node] = data  # Store the data along with node
            elif "factor" in node:
                factors.add(node)
            else:
                print(f"Warning: Node {node} has no 'type' attribute. Assuming it's a var.") # Default to var if no type
                # vars_nodes.add(node)
                vars_nodes[node] = data
        for u, v, data in graph.edges(data=True):
            # print(u, v, data)
            if u in vars_nodes and v in factors:
                var_factor_edges.append((u,v,data))
            elif v in vars_nodes and u in factors:
                var_factor_edges.append((v,u,data)) # Make sure var is the first element


        projected_graph = nx.Graph()
        # projected_graph.add_nodes_from(vars_nodes)
        # Add vars and their data to the new graph
        for var_node, data in vars_nodes.items():
            projected_graph.add_node(var_node, **data)  # Add node with its data

        # Connect vars that were connected through factors
        for i in range(len(var_factor_edges)):
            for j in range(i+1, len(var_factor_edges)):
                var1, factor1, data1 = var_factor_edges[i]
                var2, factor2, data2 = var_factor_edges[j]

                if factor1 == factor2: # Same factor, connect the vars
                    projected_graph.add_edge(var1, var2, **data1) # Add edge attributes if they exist

        labels = {}
        print(projected_graph.nodes(data=True))

        for node, data in projected_graph.nodes(data=True):
            # Check if 'label' key exists in the node's data
            if 'label' in data:
                labels[node] = data['label']
            else:
                # Fallback: use node name if no label in metadata
                labels[node] = str(node)  # Convert node to string if needed

        pos = nx.drawing.nx_agraph.graphviz_layout(projected_graph) # Use graphviz for better layout if available


        import metis
        result, partition = metis.part_graph(projected_graph, n_partitions, recursive=True)
        # print(partition)
        # # print(result)
        # print(len(projected_graph.nodes()))
        # print(len(partition))

        cmap = plt.cm.get_cmap('viridis', max(partition) + 1)  # Color map
        # nx.draw(graph, pos, with_labels=True, labels=labels, node_size=700, node_color="skyblue", font_size=10, font_color="black") # Adjust aesthetics as needed

        pos = nx.spring_layout(graph)  # You can try other layouts too
        nx.draw(projected_graph, node_color=[cmap(i) for i in partition], with_labels=True, labels=labels, node_size=700, font_size=10)
        # # plt.title(title)
        plt.show()

        # nx.draw(projected_graph, pos, with_labels=True, labels=labels, node_size=700, node_color="skyblue", font_size=10, font_color="black") # Adjust aesthetics as needed

        # # # plt.title("Graph from .dot file")
        # plt.show()
        # print(projected_graph.nodes(data=True))

        return projected_graph

    except FileNotFoundError:
        print(f"Error: .dot file not found at {dot_filepath}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def visualize_partition_overlap(graph, partition):
    """Visualizes the graph with partition colors and highlights overlap nodes.

    Args:
        graph: The NetworkX graph.
        partition: A list of integers representing the partition assignment for each node.
    """
    try:
        num_partitions = max(partition) + 1  # Get the number of partitions
        cmap = plt.cm.get_cmap('viridis', num_partitions)  # Color map for partitions

        # Find overlap nodes (nodes connected to different partitions)
        overlap_nodes = set()
        for u, v in graph.edges():
            if partition[list(graph.nodes()).index(u)] != partition[list(graph.nodes()).index(v)]:
                overlap_nodes.add(u)
                overlap_nodes.add(v)

        labels = {}
        for node, data in graph.nodes(data=True):
            # Check if 'label' key exists in the node's data
            if 'label' in data:
                labels[node] = data['label']
            else:
                # Fallback: use node name if no label in metadata
                labels[node] = str(node)  # Convert node to string if needed

        pos = nx.spring_layout(graph)  # Or any other layout you prefer

        # Draw nodes with partition colors
        nx.draw_networkx_nodes(graph, pos, nodelist=graph.nodes(),
                               node_color=[cmap(i) for i in partition],
                               node_size=700, alpha=0.8)

        # Highlight overlap nodes
        nx.draw_networkx_nodes(graph, pos, nodelist=overlap_nodes,
                               node_color="red",  # Or any color you like for overlap
                               node_size=700, alpha=1.0) # Overlap nodes are fully opaque

        nx.draw_networkx_edges(graph, pos, alpha=0.5) # Draw the edges

        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10) # Draw the labels

        plt.title("Graph Partition with Overlap Highlighted")
        plt.show()

    except Exception as e:
        print(f"An error occurred during visualization: {e}")


# draw_graph_from_dot(path)
graph = read_dot_and_project(path)
import metis
result, partition = metis.part_graph(graph, n_partitions,recursive=True)
visualize_partition_overlap(graph, partition)
# G = nx.Graph(nx.nx_pydot.read_dot(path))
# print(G.nodes(data=True))
# # G.nodes
# # print(len(G.nodes))
# # print(G.nodes.values()[0])
# # print(G.nodes[G.nodes()[0]])
# nx.draw(G, with_labels=True)
# plt.show()
