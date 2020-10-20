import neuroglancer
import numpy as np
import networkx as nx

import random


class SkeletonSource(neuroglancer.skeleton.SkeletonSource):
    def __init__(self, cc, dimensions, voxel_size, node_attrs=None, edge_attrs=None):
        super(SkeletonSource, self).__init__(dimensions)
        self.vertex_attributes["distance"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes["node_edge"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes["edge_len"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes["component"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.vertex_attributes[
            "connected_component"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.vertex_attributes[
            "component_size"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.cc = cc
        self.voxel_size = voxel_size
        for node, attrs in self.cc.nodes.items():
            assert "location" in attrs

        self.component_ids = {}
        self.connected_component_ids = {}

        for i, component in enumerate(nx.connected_components(self.cc)):
            for node in component:
                self.cc.nodes[node]["connected_component_id"] = i

    def get_skeleton(self, i):

        edges = []
        distances = []
        vertex_positions = []
        node_edge = []
        edge_len = []
        component = []
        component_size = []
        connected_component = []

        print(
            f"rendering nodes and edges with {len(self.cc.nodes)} nodes and {len(self.cc.edges)} edges"
        )

        for i, n in enumerate(self.cc.nodes):
            vertex_positions.append(
                self.cc.nodes[n]["location"] / self.voxel_size
            )
            vertex_positions.append(
                self.cc.nodes[n]["location"] / self.voxel_size
            )
            distances.append(0.1)
            distances.append(0.1)
            edges.append(2 * i)
            edges.append(2 * i + 1)
            node_edge.append(1)
            node_edge.append(1)
            edge_len.append(0)
            edge_len.append(0)
            component.append(
                self.component_ids.setdefault(
                    self.cc.nodes[n].get("component_id", 0),
                    (
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                    ),
                )
            )
            component.append(
                self.component_ids.setdefault(
                    self.cc.nodes[n].get("component_id", 0),
                    (
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                    ),
                )
            )
            connected_component.append(
                self.component_ids.setdefault(
                    self.cc.nodes[n].get("component_id", 0),
                    (
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                    ),
                )
            )
            connected_component.append(
                self.connected_component_ids.setdefault(
                    self.cc.nodes[n].get("connected_component_id", 0),
                    (
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                    ),
                )
            )
            component_size.append(self.cc.nodes[n].get("component_size", 0))
            component_size.append(self.cc.nodes[n].get("component_size", 0))
        i += 1

        for j, (u, v) in enumerate(self.cc.edges):
            vertex_positions.append(
                self.cc.nodes[u]["location"] / self.voxel_size
            )
            vertex_positions.append(
                self.cc.nodes[v]["location"] / self.voxel_size
            )
            distances.append(self.cc.edges[(u, v)].get("distance", 0.5))
            distances.append(self.cc.edges[(u, v)].get("distance", 0.5))
            edges.append((2 * i) + 2 * j)
            edges.append((2 * i) + 2 * j + 1)
            node_edge.append(0)
            node_edge.append(0)
            edge_len.append(np.linalg.norm(vertex_positions[-1] - vertex_positions[-2]))
            edge_len.append(np.linalg.norm(vertex_positions[-1] - vertex_positions[-2]))
            component.append(
                self.component_ids.setdefault(
                    self.cc.nodes[u].get("component_id", 0),
                    (
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                    ),
                )
            )
            component.append(
                self.component_ids.setdefault(
                    self.cc.nodes[v].get("component_id", 0),
                    (
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                    ),
                )
            )
            connected_component.append(
                self.connected_component_ids.setdefault(
                    self.cc.nodes[u].get("connected_component_id", 0),
                    (
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                    ),
                )
            )
            connected_component.append(
                self.connected_component_ids.setdefault(
                    self.cc.nodes[v].get("connected_component_id", 0),
                    (
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                    ),
                )
            )
            component_size.append(self.cc.nodes[u].get("component_size", 0))
            component_size.append(self.cc.nodes[v].get("component_size", 0))

        return neuroglancer.skeleton.Skeleton(
            vertex_positions=vertex_positions,
            edges=edges,
            vertex_attributes=dict(
                distance=np.array(distances),
                node_edge=node_edge,
                edge_len=edge_len,
                component=component,
                component_size=component_size,
                connected_component=connected_component,
            ),
            # edge_attribues=dict(distances=distances),
        )


class MatchSource(neuroglancer.skeleton.SkeletonSource):
    def __init__(self, graph, dimensions, voxel_size):
        super(MatchSource, self).__init__(dimensions)
        self.vertex_attributes["source"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.vertex_attributes["gt_edge"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes[
            "success_edge"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes[
            "false_match"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.vertex_attributes[
            "false_match_edge"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes["merge"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.vertex_attributes[
            "merge_edge"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes["split"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.vertex_attributes[
            "split_edge"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes["node_edge"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes[
            "selected_edge"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes[
            "component_matchings"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.vertex_attributes[
            "all_colors"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.graph = graph
        self.voxel_size = voxel_size

        self.init_colors()

    def init_colors(self):
        red = np.array((255, 128, 128)) / 256
        green = np.array((128, 255, 128)) / 256
        blue = np.array((128, 128, 255)) / 256
        yellow = np.array((255, 255, 128)) / 256
        purple = np.array((255, 128, 255)) / 256
        grey = np.array((125, 125, 125)) / 256
        cyan = np.array((0, 255, 255)) / 256
        self.error_vis_colors = {
            "split": red,
            "merge": blue,
            "true_pos": green,
            "false_neg": red,
            "false_pos": blue,
            "filtered": grey,
            "other": cyan,
        }
        self.error_vis_type = {
            (1, 0, 0, 0, 0, 0): "filtered",
            (0, 1, 0, 0, 0, 0): "true_pos",
            (0, 0, 1, 0, 0, 0): "false_pos",
            (0, 0, 0, 1, 0, 0): "false_neg",
            (0, 0, 0, 0, 1, 0): "merge",
            (0, 0, 0, 0, 0, 1): "split",
        }
        self.source_colors = np.array([(255, 128, 128), (128, 128, 255)]) / 256
        self.distance_color_range = np.array([(255, 128, 128), (128, 128, 255)]) / 256
        self.matching_colors = np.array([(255, 128, 128), (128, 128, 255)]) / 256
        self.false_colors = np.array([(255, 128, 128), (128, 128, 255)]) / 256
        self.merge_colors = np.array([(255, 128, 128), (128, 128, 255)]) / 256
        self.split_colors = np.array([(255, 128, 128), (128, 128, 255)]) / 256

    def random_color(self):
        return np.array([random.randrange(256) / 256 for _ in range(3)])

    def is_no_match_label(self, label):
        return (
            label == max(max(self.labels_y.values()), max(self.labels_x.values())) + 1
        )

    def is_no_match_node(self, node):
        return node == max(max(self.nodes_x), max(self.nodes_y)) + 1

    def get_skeleton(self, i):

        # edges only show up if both of their end points satisfy the condition
        # in shader. i.e. *_edge > 0.5

        # gt_edge, an edge from mst to mst or gt to gt
        #   simple 1 node per node
        # distance_edge, an edge in mst, labeled with color
        #   needs individual nodes per edge to annotate with distance
        # matching edge, an edge in mst or gt labeled with color
        #   edge in mst or gt where labels same on either endpoint
        # false_match_edge, self edge on mst or gt without label match
        # merge_edge, edge in mst connecting two nodes matched to different labels
        # split_edge, edge in gt connecting two nodes matched to different labels

        edges = []
        node_edge = []
        vertex_positions = []
        success_edge = []
        source = []
        gt_edge = []
        false_match = []
        false_match_edge = []
        merge = []
        merge_edge = []
        split = []
        split_edge = []
        selected_edge = []
        component_matchings = []

        all_colors = []

        component_colors = {}

        threshold_index = 25

        for i, ((u, v), attrs) in enumerate(self.graph.edges.items()):
            node_edge.append(0)
            node_edge.append(0)

            selected, success, e_fp, e_fn, e_merge, e_split, e_gt = [
                int(x) for x in attrs["details"][threshold_index]
            ]

            selected_edge.append(selected)
            selected_edge.append(selected)

            success_edge.append(success)
            success_edge.append(success)

            component_matchings.append(
                component_colors.setdefault(
                    tuple(attrs["label_pair"][threshold_index]), self.random_color()
                )
            )
            component_matchings.append(
                component_colors.setdefault(
                    tuple(attrs["label_pair"][threshold_index]), self.random_color()
                )
            )

            assert not (success and e_merge)
            assert not (success and e_split)

            # assert (
            #     sum([1 - selected, success, e_fp, e_fn, e_merge, e_split]) == 1
            # ), f"{[1 - selected, success, e_fp, e_fn, e_merge, e_split]}"
            error = (1 - selected, success, e_fp, e_fn, e_merge, e_split)
            try:
                error_vis_color = self.error_vis_colors[
                    self.error_vis_type.get(error, "other")
                ]
            except KeyError:

                selected, success, u_fp, u_fn, u_merge, u_split, u_gt = [
                    int(x) for x in attrs["details"][threshold_index]
                ]
                raise KeyError(
                    f"error (filtered, success, fp, fn, merge, split): {error}"
                )

            all_colors.append(error_vis_color)
            all_colors.append(error_vis_color)

            # from index
            u_edge = 2 * i
            v_edge = 2 * i + 1
            # from graph_attr
            u_vertex_position = self.graph.nodes[u]["location"] / tuple(self.voxel_size)
            v_vertex_position = self.graph.nodes[v]["location"] / tuple(self.voxel_size)
            # both from mst
            u_source = self.source_colors[e_gt]
            v_source = self.source_colors[e_gt]
            # yes
            u_gt_edge = e_gt
            v_gt_edge = e_gt
            # no, has to be done on a per node basis
            u_false = self.false_colors[e_fn]
            v_false = self.false_colors[e_fn]
            u_false_edge = e_fp or e_fn
            v_false_edge = e_fp or e_fn
            # yes if both match to different labels
            u_merge_edge = e_merge
            v_merge_edge = e_merge
            u_merge = self.merge_colors[e_merge]
            v_merge = self.merge_colors[e_merge]
            # no
            u_split = self.split_colors[e_split]
            v_split = self.split_colors[e_split]
            u_split_edge = e_split
            v_split_edge = e_split

            edges.append(u_edge)
            edges.append(v_edge)
            vertex_positions.append(u_vertex_position)
            vertex_positions.append(v_vertex_position)
            source.append(u_source)
            source.append(v_source)
            gt_edge.append(u_gt_edge)
            gt_edge.append(v_gt_edge)
            false_match.append(u_false)
            false_match.append(v_false)
            false_match_edge.append(u_false_edge)
            false_match_edge.append(v_false_edge)
            merge.append(u_merge)
            merge.append(v_merge)
            merge_edge.append(u_merge_edge)
            merge_edge.append(v_merge_edge)
            split.append(u_split)
            split.append(v_split)
            split_edge.append(u_split_edge)
            split_edge.append(v_split_edge)
        i += 1

        for j, (n, attrs) in enumerate(self.graph.nodes.items()):
            node_edge.append(1)
            node_edge.append(1)

            selected, success, n_fp, n_fn, n_merge, n_split, n_gt = [
                int(x) for x in attrs["details"][threshold_index]
            ]

            if selected and n_gt:
                assert success or n_fn or n_split
            elif selected:
                assert success or n_fp or n_merge

            selected_edge.append(selected)
            selected_edge.append(selected)

            success_edge.append(success)
            success_edge.append(success)

            component_matchings.append(
                component_colors.setdefault(
                    tuple(attrs["label_pair"][threshold_index]), self.random_color()
                )
            )
            component_matchings.append(
                component_colors.setdefault(
                    tuple(attrs["label_pair"][threshold_index]), self.random_color()
                )
            )

            assert not (success and n_merge)
            assert not (success and n_split)

            # assert (
            #     sum([1 - selected, success, n_fp, n_fn, n_merge, n_split]) == 1
            # ), f"{[1 - selected, success, n_fp, n_fn, n_merge, n_split]}"
            error_vis_color = self.error_vis_colors[
                self.error_vis_type[
                    (1 - selected, success, n_fp, n_fn, n_merge, n_split)
                ]
            ]
            all_colors.append(error_vis_color)
            all_colors.append(error_vis_color)

            # from index
            u_edge = 2 * (i + j)
            v_edge = 2 * (i + j) + 1
            # from graph_attr
            u_vertex_position = self.graph.nodes[n]["location"] / tuple(self.voxel_size)
            v_vertex_position = self.graph.nodes[n]["location"] / tuple(self.voxel_size)
            # both from mst
            u_source = self.source_colors[n_gt]
            v_source = self.source_colors[n_gt]
            # yes
            u_gt_edge = n_gt
            v_gt_edge = n_gt

            u_false = self.false_colors[n_fn]
            v_false = self.false_colors[n_fn]
            u_false_edge = n_fp or n_fn
            v_false_edge = n_fp or n_fn
            # yes if both match to different labels
            u_merge_edge = n_merge
            v_merge_edge = n_merge
            u_merge = self.merge_colors[n_merge]
            v_merge = self.merge_colors[n_merge]
            # no
            u_split = self.split_colors[n_split]
            v_split = self.split_colors[n_split]
            u_split_edge = n_split
            v_split_edge = n_split

            edges.append(u_edge)
            edges.append(v_edge)
            vertex_positions.append(u_vertex_position)
            vertex_positions.append(v_vertex_position)
            source.append(u_source)
            source.append(v_source)
            gt_edge.append(u_gt_edge)
            gt_edge.append(v_gt_edge)
            false_match.append(u_false)
            false_match.append(v_false)
            false_match_edge.append(u_false_edge)
            false_match_edge.append(v_false_edge)
            merge.append(u_merge)
            merge.append(v_merge)
            merge_edge.append(u_merge_edge)
            merge_edge.append(v_merge_edge)
            split.append(u_split)
            split.append(v_split)
            split_edge.append(u_split_edge)
            split_edge.append(v_split_edge)
        i += 1

        return neuroglancer.skeleton.Skeleton(
            vertex_positions=vertex_positions,
            edges=edges,
            vertex_attributes=dict(
                success_edge=success_edge,
                source=np.array(source),
                gt_edge=np.array(gt_edge),
                false_match=false_match,
                false_match_edge=false_match_edge,
                merge=merge,
                merge_edge=merge_edge,
                split=split,
                split_edge=split_edge,
                node_edge=node_edge,
                selected_edge=selected_edge,
                component_matchings=component_matchings,
                all_colors=all_colors,
            ),
            # edge_attribues=dict(distances=distances),
        )