def supervision_to_vad_segments(supervision):
    """
    Convert a list of Lhotse Supervision objects to a list of VAD segments (start, end).
    This effectively removes overlapping time segments from the supervision.
    """
    segments = []
    supervision = sorted(supervision, key=lambda x: x.start)
    cur_segment = (supervision[0].start, supervision[0].end)
    for s in supervision[1:]:
        if s.start > cur_segment[1]:
            segments.append(cur_segment)
            cur_segment = (s.start, s.end)
        else:
            cur_segment = (min(cur_segment[0], s.start), max(cur_segment[1], s.end))
    segments.append(cur_segment)
    return segments


def rttm_to_vad_segments(rttm_file):
    """
    Convert an RTTM file to a list of VAD segments (start, end).
    """
    segments = []
    with open(rttm_file, "r") as f:
        for line in f:
            line = line.strip()
            parts = line.split()
            segments.append((float(parts[3]), float(parts[4])))
    return segments
