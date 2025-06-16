from PIL import Image
import numpy as np
from io import BytesIO

class SurpriseManager:
    def __init__(self):
        pass

    def surprise(self, slices: list[bytes]) -> list[int]:
        """
        Reconstructs shredded document from vertical slices.

        Args:
            slices: list of byte arrays, each representing a JPEG-encoded vertical slice of the input document

        Returns:
            Predicted permutation of input slices to correctly reassemble the document.
        """

        def decode_slices(jpeg_byte_arrays):
            slices = []
            for byte_array in jpeg_byte_arrays:
                image = Image.open(BytesIO(byte_array))
                slices.append(np.array(image))
            return slices

        def edge_similarity(slice_a, slice_b, edge_width=1):
            right_edge = slice_a[:, -edge_width:]
            left_edge = slice_b[:, :edge_width]
            return np.mean((right_edge.astype(np.float32) - left_edge.astype(np.float32)) ** 2)

        def build_similarity_matrix(slices, edge_width=1):
            n = len(slices)
            sim = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        sim[i][j] = 1e6  # large cost to prevent self-loop
                    else:
                        sim[i][j] = edge_similarity(slices[i], slices[j], edge_width)
            return sim

        def nearest_neighbor_tsp_from_start(distances, start_node):
            n = len(distances)
            visited = [False] * n
            route = [start_node]
            visited[start_node] = True
            total_distance = 0

            for _ in range(1, n):
                last = route[-1]
                nearest = None
                min_dist = float('inf')
                for i in range(n):
                    if not visited[i] and distances[last][i] < min_dist:
                        min_dist = distances[last][i]
                        nearest = i
                route.append(nearest)
                visited[nearest] = True
                total_distance += min_dist

            return route, total_distance

        def best_tsp_route(distances):
            n = len(distances)
            best_route = None
            best_cost = float('inf')
            for start in range(n):
                route, cost = nearest_neighbor_tsp_from_start(distances, start)
                if cost < best_cost:
                    best_cost = cost
                    best_route = route
            return best_route

        decoded_slices = decode_slices(slices)
        sim_matrix = build_similarity_matrix(decoded_slices)
        order = best_tsp_route(sim_matrix)
        return order
