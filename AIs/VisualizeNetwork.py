import networkx as nx

def add_edge_to_graph(self, graph, e1, e2, c, w):
    graph.add_edge(e1, e2, color=c, weight=w)


def state_position(self, points, graph, axis):
    color_list = ['red', 'blue', 'green', 'yellow', 'purple']
    positions = {point: point for point in points}
    edges = self.G.edges()
    nodes = self.G.nodes()

    edge_colors = [self.G[u][v]['color'] for u, v in edges]
    nx.draw(graph, pos=positions, node_size=0, edge_color=edge_colors, node_color='black', ax=axis)