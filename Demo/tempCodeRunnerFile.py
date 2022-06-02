# python demo/demo_skeleton.py ${VIDEO_FILE} ${OUT_FILENAME} \
#     [--config ${SKELETON_BASED_ACTION_RECOGNITION_CONFIG_FILE}] \
#     [--checkpoint ${SKELETON_BASED_ACTION_RECOGNITION_CHECKPOINT}] \
#     [--det-config ${HUMAN_DETECTION_CONFIG_FILE}] \
#     [--det-checkpoint ${HUMAN_DETECTION_CHECKPOINT}] \
#     [--det-score-thr ${HUMAN_DETECTION_SCORE_THRESHOLD}] \
#     [--pose-config ${HUMAN_POSE_ESTIMATION_CONFIG_FILE}] \
#     [--pose-checkpoint ${HUMAN_POSE_ESTIMATION_CHECKPOINT}] \
#     [--label-map ${LABEL_MAP}] \
#     [--device ${DEVICE}] \
#     [--short-side] ${SHORT_SIDE}
from importlib.resources import path


path_output = "ResGCNv1/Demo/output"
video_path="ResGCNv1/Demo/video_fall_action_1.mp4"
path_video="{}/{}".format(path_output,video_path.split("/")[-1])
print(path_video)