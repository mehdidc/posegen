from scoreml.detection import Box, ThingOnFrame, BoxList
from scoreml.detection.match import iou_all_pairs
# https://github.com/bochinski/iou-tracker/blob/master/iou_tracker.py

def iou_tracker(objs_frames, obj_confidence_threshold=0.0, track_confidence_threshold=0.5, iou_threshold=0.5, track_length_min=2):
    tracks_active = []
    tracks_finished = []
    for frame_id, objs_frame in enumerate(objs_frames):
        objs_frame = [obj for obj in objs_frame if obj.confidence >= obj_confidence_threshold]
        updated_tracks = []
        for track in tracks_active:
            if len(objs_frame):
                src_boxes = BoxList([obj.box for obj in objs_frame]).as_array()
                target_box = track.objs[-1].box.as_array().reshape((1, 4))
                ious = iou_all_pairs(src_boxes, target_box)
                best_match = ious[:, 0].argmax()
                best_match_iou = ious.max()
                if best_match_iou >= iou_threshold:
                    best_match_score = objs_frame[best_match].confidence
                    track.max_score = max(track.max_score, best_match_score)
                    thing = objs_frame[best_match]
                    thing.frame_id = frame_id
                    track.objs.append(thing)
                    updated_tracks.append(track)
                    del objs_frame[best_match]
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                if track.max_score >= track_confidence_threshold and len(track.objs) >= track_length_min:
                    tracks_finished.append(track.objs)
        new_tracks = []
        for obj in objs_frame:
            track = Track()
            obj.frame_id = frame_id
            track.objs = [obj]
            track.max_score = obj.confidence
            new_tracks.append(track)
        tracks_active = updated_tracks + new_tracks
    
    tracks_finished += [
        track.objs
        for track in tracks_active
        if track.max_score >= track_confidence_threshold and len(track.objs) >= track_length_min
    ]
    return tracks_finished


class Track:

    def __init__(self):
        self.objs = []
        self.max_score = 0
