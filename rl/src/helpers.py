import numpy as np

from gridworldtypes import Direction, Wall


# helpers
def manhattan(arr1: np.ndarray, arr2: np.ndarray):
    """Gets the manhattan distance between two locations"""
    return np.sum(np.abs(arr1 - arr2))


# bitwise operation helpers
def get_bit(value: int, bit_index: int):
    """Get the value at a particular bit of the provided int"""
    return (value >> bit_index) & 1


def rotate_right(arr: np.ndarray, shift: int, bit_width: int = 8):
    """Bitwise shift with rotation"""
    return ((arr >> shift) | (arr << (bit_width - shift))) & ((1 << bit_width) - 1)


def zero_bit(arr: np.ndarray, bit_index: int):
    """Zeroes out the particular bit of all values in the given array"""
    mask = np.uint8(1) << bit_index
    return arr & ~mask


# viewcone / world / numpy idx conversions
def view_to_world(
    agent_loc: np.ndarray, agent_dir: Direction, view_coord: np.ndarray
) -> np.ndarray:
    """
    Maps viewcone coordinate to world coordinate
    """
    dirrect = Direction(agent_dir)
    # match Direction(agent_dir):
    if dirrect == Direction.RIGHT:
            return agent_loc + view_coord
    if dirrect == Direction.DOWN:
            return agent_loc - np.array((view_coord[1], -view_coord[0]))
    if dirrect == Direction.LEFT:
            return agent_loc - view_coord
    if dirrect == Direction.UP:
            return agent_loc + np.array((view_coord[1], -view_coord[0]))


def world_to_view(
    agent_loc: np.ndarray, agent_dir: Direction, world_coord: np.ndarray
) -> np.ndarray:
    """
    Maps world coordinate to viewcone coordinate
    """
    dirrect = Direction(agent_dir)
    # match Direction(agent_dir):
    if dirrect == Direction.RIGHT:
            return world_coord - agent_loc
    if dirrect == Direction.DOWN:
            return np.array(
                ((world_coord[1] - agent_loc[1]), (agent_loc[0] - world_coord[0]))
            )
    if dirrect == Direction.LEFT:
            return agent_loc - world_coord
    if dirrect == Direction.UP:
            return np.array(
                ((agent_loc[1] - world_coord[1]), (world_coord[0] - agent_loc[0]))
            )


def idx_to_view(
    idx: tuple[int, int], viewcone: tuple[int, int, int, int]
) -> np.ndarray:
    """Converts numpy array index to viewcone index"""
    return np.array(idx) - np.array((viewcone[0], viewcone[2]))


def view_to_idx(
    view_coord: np.ndarray, viewcone: tuple[int, int, int, int]
) -> tuple[int, int]:
    """Converts viewcone index to numpy array index"""
    # using .item() to return a native python int, mostly just for neater debug printing
    return (view_coord[0].item() + viewcone[0], view_coord[1].item() + viewcone[2])


def is_idx_valid(idx: np.ndarray, viewcone_length: int, viewcone_width: int):
    """Checks if a numpy index value is within the viewcone"""
    return (0 <= idx[0] < viewcone_length) and (0 <= idx[1] < viewcone_width)


def is_world_coord_valid(world_coord: np.ndarray, size: int):
    """Checks if a world coordinate value is within the world"""
    return (0 <= world_coord[0] < size) and (0 <= world_coord[1] < size)


def supercover_line(start: np.ndarray, end: np.ndarray) -> list[tuple[int, int]]:
    """Returns a list of (x, y) tiles the line passes through."""
    x0, y0 = start[0], start[1]
    x1, y1 = end[0], end[1]
    tiles = []

    # get x and y distance you need to pass through
    dx = x1 - x0
    dy = y1 - y0

    nx = abs(dx)
    ny = abs(dy)

    sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
    sign_y = 1 if dy > 0 else -1 if dy < 0 else 0

    # starting tile
    px, py = x0.item(), y0.item()
    tiles.append((px, py))

    # distance travelled so far on each axis, ix and iy
    ix = iy = 0
    while ix < nx or iy < ny:
        # check for diagonal tile case
        # fixes asymmetric tile visibility around perfectly diagonal corners
        if (1 + 2 * ix) * ny == (1 + 2 * iy) * nx:
            px += sign_x
            py += sign_y
            ix += 1
            iy += 1
        elif (1 + 2 * ix) * ny < (1 + 2 * iy) * nx:
            px += sign_x
            ix += 1
        else:
            py += sign_y
            iy += 1
        tiles.append((px, py))

    return tiles


# maze generation
def convert_tile_to_edge(arena: np.ndarray, grid: np.ndarray):
    """
    converts a numpy grid maze setup as generated by mazelib to the edge-centric style the env expects,
    and modifies the input `arena` parameter inplace
    """
    _grid = grid.astype(dtype=np.uint8)
    # turn all the outside edges into the appropriate walls
    arena += _grid[1::2, 0:-2:2] * Wall.TOP.power
    arena += _grid[1::2, 2::2] * Wall.BOTTOM.power
    arena += _grid[0:-2:2, 1::2] * Wall.LEFT.power
    arena += _grid[2::2, 1::2] * Wall.RIGHT.power
