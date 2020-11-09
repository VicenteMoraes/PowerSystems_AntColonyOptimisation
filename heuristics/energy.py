import heuristics.methods as methods

def euclideanEnergy(node1, node2, weight=1, squared=True):
    if squared:
        return weight * methods.euclideanDistanceSquared(node1, node2)
    else:
        return weight * (methods.euclideanDistanceSquared(node1, node2) ** 0.5)