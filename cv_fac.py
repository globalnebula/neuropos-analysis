# cv_fac.py
import mediapipe as mp
import numpy as np
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ========== UTILS ==========
def calc_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def calc_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c.y - b.y, c.x - b.x) -
        math.atan2(a.y - b.y, a.x - b.x))
    return abs(angle)

def get_metrics(landmarks):
    l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    stride_length = calc_distance(l_ankle, r_ankle)
    step_width = abs(l_ankle.x - r_ankle.x)
    knee_symmetry = abs(l_knee.y - r_knee.y)
    arm_swing_left = calc_distance(l_shoulder, l_wrist)
    arm_swing_right = calc_distance(r_shoulder, r_wrist)
    torso_angle = calc_angle(l_shoulder, r_shoulder, r_hip)
    pelvic_drop = abs(l_hip.y - r_hip.y)

    # Extra metrics
    knee_angle_left = calc_angle(l_hip, l_knee, l_ankle)
    knee_angle_right = calc_angle(r_hip, r_knee, r_ankle)
    elbow_angle_left = calc_angle(l_shoulder, l_elbow, l_wrist)
    elbow_angle_right = calc_angle(r_shoulder, r_elbow, r_wrist)
    shoulder_width = calc_distance(l_shoulder, r_shoulder)
    hip_width = calc_distance(l_hip, r_hip)
    shoulder_hip_ratio = shoulder_width / hip_width if hip_width != 0 else 0
    upper_body_posture = calc_angle(nose, l_shoulder, r_shoulder)

    return {
        "stride_length": float(stride_length),
        "step_width": float(step_width),
        "knee_symmetry": float(knee_symmetry),
        "arm_swing_left": float(arm_swing_left),
        "arm_swing_right": float(arm_swing_right),
        "torso_angle": float(torso_angle),
        "pelvic_drop": float(pelvic_drop),
        "knee_angle_left": float(knee_angle_left),
        "knee_angle_right": float(knee_angle_right),
        "elbow_angle_left": float(elbow_angle_left),
        "elbow_angle_right": float(elbow_angle_right),
        "shoulder_width": float(shoulder_width),
        "hip_width": float(hip_width),
        "shoulder_hip_ratio": float(shoulder_hip_ratio),
        "upper_body_posture": float(upper_body_posture)
    }

# ========== CORE STREAMLIT CALLBACK ==========
metrics_log = []

def process_frame(image, return_metrics=False):
    import cv2
    from mediapipe.python.solutions import drawing_utils as mp_drawing

    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    live_metrics = {}

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        metrics = get_metrics(results.pose_landmarks.landmark)
        metrics_log.append(metrics)
        live_metrics = metrics

        y = 20
        for k, v in metrics.items():
            cv2.putText(image, f"{k}: {v:.4f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y += 18

    if return_metrics:
        return image, live_metrics
    else:
        return image

def get_average_metrics():
    if not metrics_log:
        return {}
    return {k: float(np.mean([m[k] for m in metrics_log if k in m])) for k in metrics_log[0]}

def get_live_metrics():
    if not metrics_log:
        return {}
    return metrics_log[-1]