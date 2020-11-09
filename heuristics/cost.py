import heuristics.methods as methods

def euclideanCost(graph, weight=1, squared=True):
    if squared:
        return [[weight * methods.euclideanDistanceSquared(x, y) for x in graph.nodes] for y in graph.nodes]
    else:
        return [[weight * (methods.euclideanDistanceSquared(x, y) ** 0.5) for x in graph.nodes] for y in graph.nodes]