import neuroglancer
import numpy as np
import networkx as nx

from .graph_source import SkeletonSource

import logging
import copy

logger = logging.getLogger(__name__)


def add_trees_no_skeletonization(
    s, trees, node_id, name, dimensions, visible=False, color=None
):
    mst = []
    for u, v in trees.edges():
        pos_u = np.array(trees.nodes[u]["location"]) + 0.5
        pos_v = np.array(trees.nodes[v]["location"]) + 0.5
        mst.append(
            neuroglancer.LineAnnotation(
                point_a=pos_u[::-1], point_b=pos_v[::-1], id=next(node_id)
            )
        )

    s.layers.append(
        name="{}".format(name),
        layer=neuroglancer.AnnotationLayer(annotations=mst),
        annotationColor="#{:02X}{:02X}{:02X}".format(255, 125, 125),
        visible=visible,
    )


def add_graph(
    s,
    graph: nx.Graph,
    name: str,
    graph_dimensions,
    visible=False,
):

    array_dimensions = copy.deepcopy(graph_dimensions)
    offset = np.min([attrs["location"] for attrs in graph.nodes.values()], axis=0)
    voxel_size = (1,) * len(offset)

    s.layers.append(
        name="{}".format(name),
        layer=neuroglancer.SegmentationLayer(
            source=[
                neuroglancer.LocalVolume(
                    data=np.ones((1, 1, 1), dtype=np.uint32),
                    dimensions=array_dimensions,
                    voxel_offset=offset,
                ),
                SkeletonSource(graph, graph_dimensions, voxel_size=voxel_size),
            ],
            skeleton_shader="""
#uicontrol float showautapse slider(min=0, max=2)

void main() {
    if (distance > showautapse) discard;
    emitRGB(colormapJet(distance));
}
""",
            selected_alpha=0,
            not_selected_alpha=0,
        ),
    )


def visualize_graph(graph: nx.Graph, name: str = "graph", dimensions=None):
    if dimensions is None:
        node, attrs = next(iter(graph.nodes.items()))
        loc = attrs["location"]
        n_dims = len(loc)
        dims = ["t", "z", "y", "x"][-n_dims:]
        units = "nm"
        scales = (1,) * n_dims
        attrs = {"names": dims, "units": units, "scales": scales}
        dimensions = neuroglancer.CoordinateSpace(**attrs)

    viewer = neuroglancer.Viewer()
    viewer.dimensions = dimensions
    with viewer.txn() as s:
        add_graph(
            s,
            graph,
            name=name,
            visible=True,
            graph_dimensions=dimensions,
        )
    print(viewer)
    input("Hit ENTER to quit!")
