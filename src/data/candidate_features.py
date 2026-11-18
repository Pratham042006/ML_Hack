"""Optional candidate-relative and directed-position features for model scoring."""
from collections import defaultdict
import numpy as np
from src.data.candidate_projection import project_candidate_positions
from src.data.heading import HeadingConfig, add_heading_features
from src.data.pipeline import _project_local


def add_candidate_features(trip, candidates, config=HeadingConfig()):
    """Return copies with six model features; never read reference routes."""
    candidates = add_heading_features(trip, candidates, config)
    if not candidates:
        return []
    fractions = project_candidate_positions(trip, candidates)
    groups = defaultdict(list)
    for c in candidates:
        groups[c['observation_index']].append(c)
    ranks, closest = {}, {}
    for obs, rows in groups.items():
        ordered = sorted(rows, key=lambda c:(c['features']['distance_m'], c['arc_id']))
        closest[obs] = ordered[0]['features']['distance_m']
        for i, c in enumerate(ordered, 1): ranks[(obs, c['arc_id'])] = i
    nodes = trip['network']['nodes']
    metric = _project_local(list(nodes) + [[o['longitude'], o['latitude']] for o in trip['observations']])
    arcs = trip['network']['arcs']
    lengths = {}
    for c in candidates:
        arc = c['arc_id']
        if arc not in lengths:
            a, b = arcs[arc]
            lengths[arc] = float(np.linalg.norm(metric[b]-metric[a]))
    return [{**c, 'features': {**c['features'],
             'distance_margin_m': c['features']['distance_m'] - closest[c['observation_index']],
             'projection_fraction': fractions[(c['observation_index'], c['arc_id'])],
             'candidate_rank': ranks[(c['observation_index'], c['arc_id'])],
             'segment_length_m': lengths[c['arc_id']]}}
            for c in candidates]
