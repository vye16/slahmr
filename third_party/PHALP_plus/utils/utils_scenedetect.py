from scenedetect.scene_manager import SceneManager
from scenedetect.scene_detector import SceneDetector
from typing import List, Optional, Tuple
from scenedetect.frame_timecode import FrameTimecode
from scenedetect import open_video

def detect(video_path: str,
           detector: SceneDetector,
           stats_file_path: Optional[str] = None,
           show_progress: bool = False) -> List[Tuple[FrameTimecode, FrameTimecode]]:
    """Perform scene detection on a given video `path` using the specified `detector`.

    Arguments:
        video_path: Path to input video (absolute or relative to working directory).
        detector: A `SceneDetector` instance (see :py:mod:`scenedetect.detectors` for a full list
            of detectors).
        stats_file_path: Path to save per-frame metrics to for statistical analysis or to
            determine a better threshold value.
        show_progress: Show a progress bar with estimated time remaining. Default is False.

    Returns:
        List of scenes (pairs of :py:class:`FrameTimecode` objects).

    Raises:
        :py:class:`VideoOpenFailure`: `video_path` could not be opened.
        :py:class:`StatsFileCorrupt`: `stats_file_path` is an invalid stats file
    """
    video = open_video(video_path)
    if stats_file_path:
        scene_manager = SceneManager(StatsManager())
    else:
        scene_manager = SceneManager()
    scene_manager.add_detector(detector)
    scene_manager.detect_scenes(video=video, show_progress=show_progress)
    #if not scene_manager.stats_manager is None:
    #    scene_manager.stats_manager.save_to_csv(
    #        csv_file=stats_file_path, base_timecode=video.base_timecode)
    return scene_manager.get_scene_list()
