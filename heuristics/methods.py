import ACO

def euclideanDistanceSquared(node1: ACO.Node, node2: ACO.Node):
    return sum((x1-x2) ** 2 for x1, x2 in zip(node1.position, node2.position))
