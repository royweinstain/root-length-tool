"""Measure arc-length along root skeletons between marks."""

import cv2
import numpy as np
from collections import defaultdict
from scipy.ndimage import distance_transform_edt

from .utils import direction_to_vector

# A mark is associated with a root only if it lies within this many pixels of
# that root's skeleton. Marks farther than this are left unassigned.
MARK_MAX_DIST = 15


def assign_marks_to_roots(skeleton, mark_mask, labels, max_dist=MARK_MAX_DIST):
    """Assign each physical mark to exactly ONE root — its nearest skeleton.

    Each connected component of ``mark_mask`` is one physical mark. It is
    assigned to the root whose skeleton is closest to it (by the mark's closest
    pixel), provided that distance is within ``max_dist``; otherwise the mark is
    left unassigned (root_id 0). This avoids the previous behaviour where a mark
    near two roots could be counted on both, and makes association explicit so
    it can be displayed.

    Args:
        skeleton: Boolean skeleton image.
        mark_mask: Binary mask of detected marks (0/255).
        labels: Labeled image of root components (root id per pixel).
        max_dist: Maximum mark-to-skeleton distance (px) to allow assignment.

    Returns:
        (mark_labels, assignments) where mark_labels is the labeled mark image
        and assignments is a list of dicts:
            {"mark_label": int, "centroid": (y, x),
             "root_id": int (0 = unassigned), "snapped": (y, x) or None}
    """
    num_marks, mark_labels = cv2.connectedComponents(mark_mask)
    assignments = []

    skel = skeleton.astype(bool)
    if not skel.any():
        for ml in range(1, num_marks):
            ys, xs = np.where(mark_labels == ml)
            assignments.append({
                "mark_label": ml,
                "centroid": (int(ys.mean()), int(xs.mean())),
                "root_id": 0, "snapped": None,
            })
        return mark_labels, assignments

    # For every pixel: distance to the nearest skeleton pixel, and that pixel's
    # coordinates (so we can read off its root label).
    dist, (iy, ix) = distance_transform_edt(~skel, return_indices=True)

    for ml in range(1, num_marks):
        ys, xs = np.where(mark_labels == ml)
        # Use the mark pixel closest to any skeleton.
        d = dist[ys, xs]
        k = int(np.argmin(d))
        centroid = (int(ys.mean()), int(xs.mean()))
        if d[k] <= max_dist:
            sy, sx = int(iy[ys[k], xs[k]]), int(ix[ys[k], xs[k]])
            root_id = int(labels[sy, sx])
            snapped = (sy, sx)
        else:
            root_id, snapped = 0, None
        assignments.append({
            "mark_label": ml, "centroid": centroid,
            "root_id": root_id, "snapped": snapped,
        })

    return mark_labels, assignments


def find_mark_positions_on_skeleton(skeleton, mark_mask, labels, num_roots,
                                    direction="down"):
    """Find each root's marks (one mark -> one nearest root) ordered along it.

    Args:
        skeleton: Boolean skeleton image.
        mark_mask: Binary mask of detected marks (0/255).
        labels: Labeled image of root components.
        num_roots: Number of roots.
        direction: Growth direction (plant body -> tip): "up"/"down"/"left"/"right".
                   Marks are ordered from the plant-body end outward.

    Returns:
        Dict mapping root_id -> list of (y, x) mark positions ordered along skeleton.
    """
    _, assignments = assign_marks_to_roots(skeleton, mark_mask, labels)

    per_root = defaultdict(list)
    for a in assignments:
        if a["root_id"] > 0 and a["snapped"] is not None:
            per_root[a["root_id"]].append(a["snapped"])

    mark_positions = {}
    for root_id, points in per_root.items():
        root_skeleton = skeleton & (labels == root_id)
        mark_positions[root_id] = _order_points_along_skeleton(
            points, root_skeleton, direction
        )

    return mark_positions


def measure_segments(mark_positions, skeleton, labels, pixels_per_mm=None,
                     include_tip=False, direction="down"):
    """Measure arc-length between consecutive marks along each root.

    Args:
        mark_positions: Dict from find_mark_positions_on_skeleton.
        skeleton: Boolean skeleton image.
        labels: Labeled root image.
        pixels_per_mm: Scale factor. If None, returns pixel distances.
        include_tip: If True, add the root tip (lowest skeleton endpoint) as
                     the final measurement point when it doesn't coincide
                     with the last mark. This handles cases where students
                     don't mark the very end of the root.

    Returns:
        Dict mapping root_id -> {
            "total_length_mm": float,
            "segments": [float, ...],
            "mark_count": int,
            "tip_included": bool,
        }
    """
    results = {}
    scale = pixels_per_mm if pixels_per_mm else 1.0

    for root_id, positions in mark_positions.items():
        root_skel = skeleton & (labels == root_id)

        # Optionally add the root tip as the final point
        tip_included = False
        if include_tip and len(positions) >= 1:
            tip = _find_root_tip(root_skel, direction)
            if tip is not None:
                last_mark = positions[-1]
                tip_dist = np.sqrt((tip[0] - last_mark[0])**2 +
                                   (tip[1] - last_mark[1])**2)
                # Only add tip if it's far enough from the last mark
                # (more than 10 pixels away = not already marked)
                if tip_dist > 10:
                    positions = positions + [tip]
                    tip_included = True

        if len(positions) < 2:
            results[root_id] = {
                "total_length_mm": 0.0,
                "segments": [],
                "mark_count": len(positions),
                "tip_included": False,
            }
            continue

        segments = []
        for i in range(len(positions) - 1):
            start = positions[i]
            end = positions[i + 1]
            dist_px = _trace_path_length(root_skel, start, end)
            dist_mm = dist_px / scale
            segments.append(round(dist_mm, 2))

        results[root_id] = {
            "total_length_mm": round(sum(segments), 2),
            "segments": segments,
            "mark_count": len(positions),
            "tip_included": tip_included,
        }

    # Include every labeled root, even those with no detected marks, so small
    # pieces (e.g. from splitting/merging) stay listed and can be identified.
    for root_id in np.unique(labels):
        rid = int(root_id)
        if rid == 0 or rid in results:
            continue
        results[rid] = {
            "total_length_mm": 0.0,
            "segments": [],
            "mark_count": 0,
            "tip_included": False,
        }

    return results


def _skeleton_endpoints(skeleton):
    """Return (ys, xs) of endpoint pixels (exactly one neighbor) in skeleton."""
    skel = skeleton.astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skel, -1, kernel)
    endpoint_mask = (skel > 0) & (neighbors == 1)
    return np.where(endpoint_mask)


def base_endpoint(skeleton, direction="down"):
    """Find the plant-body end of a root skeleton for the given growth direction.

    The body endpoint is the one that is most "upstream" — i.e. has the smallest
    projection onto the growth-direction vector. Falls back to all skeleton
    pixels if there are no clean endpoints.

    Returns (y, x) or None.
    """
    dy, dx = direction_to_vector(direction)
    ys, xs = _skeleton_endpoints(skeleton)
    if len(ys) == 0:
        ys, xs = np.where(skeleton > 0)
        if len(ys) == 0:
            return None
    proj = ys * dy + xs * dx
    idx = int(np.argmin(proj))
    return (int(ys[idx]), int(xs[idx]))


def _find_root_tip(skeleton, direction="down"):
    """Find the root tip — the endpoint farthest along the growth direction.

    The tip is the endpoint with the largest projection onto the growth-direction
    vector (plant body -> tip). Works for any orientation.

    Returns (y, x) tuple or None.
    """
    dy, dx = direction_to_vector(direction)
    ys, xs = _skeleton_endpoints(skeleton)
    if len(ys) == 0:
        return None
    if len(ys) == 1:
        return (int(ys[0]), int(xs[0]))
    proj = ys * dy + xs * dx
    max_idx = int(np.argmax(proj))
    return (int(ys[max_idx]), int(xs[max_idx]))


def _cluster_points(points, min_dist=15):
    """Simple agglomerative clustering of 2D points."""
    if len(points) == 0:
        return []

    clusters = [[p] for p in points]
    merged = True

    while merged:
        merged = False
        new_clusters = []
        used = set()

        for i in range(len(clusters)):
            if i in used:
                continue
            current = clusters[i]
            for j in range(i + 1, len(clusters)):
                if j in used:
                    continue
                # Check if any point in cluster j is close to any point in cluster i
                for p1 in current:
                    for p2 in clusters[j]:
                        if np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) < min_dist:
                            current = current + clusters[j]
                            used.add(j)
                            merged = True
                            break
                    if j in used:
                        break
            new_clusters.append(current)
            used.add(i)

        clusters = new_clusters

    return clusters


def _order_points_along_skeleton(points, skeleton, direction="down"):
    """Order mark positions along the skeleton from plant body to tip.

    Uses BFS from the plant-body endpoint (the endpoint most upstream relative
    to the growth direction) to compute the shortest-path distance to each mark,
    then sorts by distance. This handles branching skeletons and any growth
    orientation correctly.
    """
    if len(points) <= 1:
        return points

    from collections import deque

    skel = skeleton.astype(np.uint8)
    h, w = skel.shape
    ys, xs = np.where(skel > 0)
    if len(ys) == 0:
        return points

    # Anchor BFS at the plant-body endpoint for this growth direction.
    start = base_endpoint(skel, direction)
    if start is None:
        return points

    # BFS from plant body to compute distance to every skeleton pixel
    dist_map = np.full((h, w), -1.0)
    dist_map[start[0], start[1]] = 0.0
    queue = deque([start])

    while queue:
        cy, cx = queue.popleft()
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] > 0 and dist_map[ny, nx] < 0:
                    step = np.sqrt(2) if (dy != 0 and dx != 0) else 1.0
                    dist_map[ny, nx] = dist_map[cy, cx] + step
                    queue.append((ny, nx))

    # For each mark, find its distance from plant body (snap to nearest skeleton pixel)
    point_dists = []
    for py, px in points:
        snapped = _snap_to_skeleton(skel, (py, px), search_radius=10)
        if snapped is not None and dist_map[snapped[0], snapped[1]] >= 0:
            d = dist_map[snapped[0], snapped[1]]
        else:
            d = float("inf")
        point_dists.append((d, (py, px)))

    point_dists.sort(key=lambda x: x[0])
    return [p for _, p in point_dists]


def _trace_skeleton_path(skeleton):
    """Trace a skeleton to get an ordered list of pixel coordinates.

    Starts from an endpoint and follows the path.
    """
    skel = skeleton.astype(np.uint8)
    ys, xs = np.where(skel > 0)
    if len(ys) == 0:
        return []

    # Find endpoints
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skel, -1, kernel)
    endpoint_mask = (skel > 0) & (neighbors == 1)
    ep_ys, ep_xs = np.where(endpoint_mask)

    # Start from the endpoint closest to the plant body (topmost = smallest y)
    # This ensures the path is traced from plant body toward root tip
    if len(ep_ys) > 0:
        top_idx = np.argmin(ep_ys)
        start = (ep_ys[top_idx], ep_xs[top_idx])
    else:
        top_idx = np.argmin(ys)
        start = (ys[top_idx], xs[top_idx])

    # BFS/DFS trace from start
    path = []
    visited = set()
    stack = [start]

    while stack:
        cy, cx = stack.pop()
        if (cy, cx) in visited:
            continue
        visited.add((cy, cx))
        path.append((cy, cx))

        # Check 8-connected neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]:
                    if skel[ny, nx] > 0 and (ny, nx) not in visited:
                        stack.append((ny, nx))

    return path


def _trace_path_length(skeleton, start, end):
    """Measure the arc-length along skeleton between two points.

    Uses BFS to find the shortest path along the skeleton between start and end.
    Returns the path length in pixels (diagonal steps count as sqrt(2)).
    """
    skel = skeleton.astype(np.uint8)
    h, w = skel.shape

    # Find nearest skeleton pixels to start and end
    start = _snap_to_skeleton(skel, start)
    end = _snap_to_skeleton(skel, end)

    if start is None or end is None:
        return 0.0

    # BFS from start to end along skeleton
    from collections import deque
    queue = deque()
    queue.append((start, 0.0))
    visited = {start}

    while queue:
        (cy, cx), dist = queue.popleft()

        if (cy, cx) == end:
            return dist

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited:
                    if skel[ny, nx] > 0:
                        step = np.sqrt(2) if (dy != 0 and dx != 0) else 1.0
                        visited.add((ny, nx))
                        queue.append(((ny, nx), dist + step))

    # If no path found, return Euclidean distance as fallback
    return np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)


def _snap_to_skeleton(skeleton, point, search_radius=10):
    """Find the nearest skeleton pixel to a given point."""
    y, x = point
    h, w = skeleton.shape

    # Search in expanding radius
    for r in range(search_radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] > 0:
                    return (ny, nx)
    return None
