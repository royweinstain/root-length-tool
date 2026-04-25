"""Measure arc-length along root skeletons between marks."""

import cv2
import numpy as np
from collections import defaultdict


def find_mark_positions_on_skeleton(skeleton, mark_mask, labels, num_roots):
    """Find where marks intersect each root's skeleton.

    Args:
        skeleton: Boolean skeleton image.
        mark_mask: Binary mask of detected marks (0/255).
        labels: Labeled image of root components.
        num_roots: Number of roots.

    Returns:
        Dict mapping root_id -> list of (y, x) mark positions ordered along skeleton.
    """
    # Dilate marks to ensure intersection with thin skeleton
    # Radius of 15px covers marks that are near but not exactly on the skeleton
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    marks_dilated = cv2.dilate(mark_mask, kernel, iterations=1)

    # Find intersection pixels
    intersection = skeleton & (marks_dilated > 0)

    mark_positions = defaultdict(list)

    for root_id in range(1, num_roots + 1):
        root_skeleton = skeleton & (labels == root_id)
        root_intersections = intersection & (labels == root_id)

        if not np.any(root_intersections):
            continue

        # Get intersection pixel coordinates
        ys, xs = np.where(root_intersections)
        if len(ys) == 0:
            continue

        # Cluster nearby intersection points (a single mark may hit multiple skeleton pixels)
        points = np.column_stack((ys, xs))
        clusters = _cluster_points(points, min_dist=15)

        # Get the centroid of each cluster
        centroids = []
        for cluster in clusters:
            cy = int(np.mean([p[0] for p in cluster]))
            cx = int(np.mean([p[1] for p in cluster]))
            centroids.append((cy, cx))

        # Order centroids along the skeleton path
        ordered = _order_points_along_skeleton(centroids, root_skeleton)
        mark_positions[root_id] = ordered

    return dict(mark_positions)


def measure_segments(mark_positions, skeleton, labels, pixels_per_mm=None,
                     include_tip=False):
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
            tip = _find_root_tip(root_skel)
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

    return results


def _find_root_tip(skeleton):
    """Find the root tip — the endpoint farthest from the plant body.

    Roots typically grow downward or rightward. The tip is the endpoint
    farthest from the topmost/leftmost point of the skeleton (where the
    seed/plant body is).

    Returns (y, x) tuple or None.
    """
    skel = skeleton.astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skel, -1, kernel)
    endpoint_mask = (skel > 0) & (neighbors == 1)

    ys, xs = np.where(endpoint_mask)
    if len(ys) == 0:
        return None

    if len(ys) == 1:
        return (int(ys[0]), int(xs[0]))

    # The root tip is the endpoint farthest from the skeleton's center of mass
    # weighted toward the top (where the plant body is)
    all_ys, all_xs = np.where(skel > 0)
    # Plant body is near the top-left of the skeleton (min y, min x)
    body_y = np.percentile(all_ys, 10)
    body_x = np.percentile(all_xs, 10)

    # Find the endpoint farthest from the plant body
    dists = (ys - body_y)**2 + (xs - body_x)**2
    max_idx = np.argmax(dists)
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


def _order_points_along_skeleton(points, skeleton):
    """Order mark positions along the skeleton from plant body to tip.

    Uses BFS from the topmost endpoint (plant body) to compute the
    shortest-path distance to each mark, then sorts by distance.
    This handles branching skeletons correctly.
    """
    if len(points) <= 1:
        return points

    from collections import deque

    skel = skeleton.astype(np.uint8)
    h, w = skel.shape
    ys, xs = np.where(skel > 0)
    if len(ys) == 0:
        return points

    # Find the plant body endpoint (topmost skeleton endpoint)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skel, -1, kernel)
    ep_mask = (skel > 0) & (neighbors == 1)
    ep_ys, ep_xs = np.where(ep_mask)

    if len(ep_ys) > 0:
        top_idx = np.argmin(ep_ys)
        start = (int(ep_ys[top_idx]), int(ep_xs[top_idx]))
    else:
        top_idx = np.argmin(ys)
        start = (int(ys[top_idx]), int(xs[top_idx]))

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
