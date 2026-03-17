from collections import deque

def bfs(graph, start):
    """
    Performs Breadth-First Search (BFS) on a graph starting from a given node.

    :param graph: A dictionary representing the adjacency list of the graph.
    :param start: The starting node for BFS.
    :return: A list of nodes in the order they were visited.
    """
    visited = set()
    queue = deque([start])
    visited.add(start)
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order

# Example usage
if __name__ == "__main__":
    # Example graph represented as adjacency list
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }

    start_node = 'A'
    traversal_order = bfs(graph, start_node)
    print(f"BFS traversal starting from {start_node}: {traversal_order}")
