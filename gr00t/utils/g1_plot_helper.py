
MODALITY_TO_JOINTS = {
    "left_arm": ["L_SHOULDER_PITCH", "L_SHOULDER_ROLL", "L_SHOULDER_YAW", 
                 "L_ELBOW", "L_WRIST_ROLL", "L_WRIST_PITCH", "L_WRIST_YAW"],
    "right_arm": ["R_SHOULDER_PITCH", "R_SHOULDER_ROLL", "R_SHOULDER_YAW", 
                 "R_ELBOW", "R_WRIST_ROLL", "R_WRIST_PITCH", "R_WRIST_YAW"],
    "left_hand": ["L_THUMB_0", "L_THUMB_1", "L_THUMB_2", "L_MIDDLE_0", 
                  "L_MIDDLE_1", "L_INDEX_0", "L_INDEX_1"],
    "right_hand": ["R_THUMB_0", "R_THUMB_1", "R_THUMB_2", "R_MIDDLE_0", 
                   "R_MIDDLE_1", "R_INDEX_0", "R_INDEX_1"]
}

ALL_JOINT_NAMES = [
  'L_LEG_HIP_PITCH',    # 0
  'L_LEG_HIP_ROLL',     # 1
  'L_LEG_HIP_YAW',      # 2
  'L_LEG_KNEE',         # 3
  'L_LEG_ANKLE_PITCH',  # 4
  'L_LEG_ANKLE_ROLL',   # 5
  'R_LEG_HIP_PITCH',    # 6
  'R_LEG_HIP_ROLL',     # 7
  'R_LEG_HIP_YAW',      # 8
  'R_LEG_KNEE',         # 9
  'R_LEG_ANKLE_PITCH',  # 10
  'R_LEG_ANKLE_ROLL',   # 11
  'WAIST_YAW',          # 12
  'WAIST_ROLL',         # 13
  'WAIST_PITCH',      # 14
  'L_SHOULDER_PITCH', # 15
  'L_SHOULDER_ROLL',  # 16
  'L_SHOULDER_YAW',   # 17
  'L_ELBOW',          # 18
  'L_WRIST_ROLL',     # 19
  'L_WRIST_PITCH',    # 20
  'L_WRIST_YAW',      # 21
  'R_SHOULDER_PITCH', # 22
  'R_SHOULDER_ROLL',  # 23
  'R_SHOULDER_YAW',   # 24
  'R_ELBOW',          # 25
  'R_WRIST_ROLL',     # 26
  'R_WRIST_PITCH',    # 27
  'R_WRIST_YAW',      # 28
  "L_THUMB_0", 
  "L_THUMB_1", 
  "L_THUMB_2", 
  "L_MIDDLE_0",
  "L_MIDDLE_1", 
  "L_INDEX_0", 
  "L_INDEX_1",
  "R_THUMB_0", 
  "R_THUMB_1", 
  "R_THUMB_2", 
  "R_MIDDLE_0",
  "R_MIDDLE_1", 
  "R_INDEX_0", 
  "R_INDEX_1"
]

# radians
JOINT_LIMITS_LOWER = [
  -2.5307,        # L_LEG_HIP_PITCH
  -0.5236,        # L_LEG_HIP_ROLL
  -2.7576,        # L_LEG_HIP_YAW
  -0.087267,      # L_LEG_KNEE
  -0.87267,       # L_LEG_ANKLE_PITCH
  -0.2618,        # L_LEG_ANKLE_ROLL
  -2.5307,        # R_LEG_HIP_PITCH
  -2.9671,        # R_LEG_HIP_ROLL
  -2.7576,        # R_LEG_HIP_YAW
  -0.087267,      # R_LEG_KNEE
  -0.87267,       # R_LEG_ANKLE_PITCH
  -0.2618,        # R_LEG_ANKLE_ROLL
  -2.618,         # WAIST_YAW
  -0.52,          # WAIST_ROLL
  -0.52,          # WAIST_PITCH
  -3.0892,        # L_SHOULDER_PITCH
  -1.5882,        # L_SHOULDER_ROLL
  -2.618,         # L_SHOULDER_YAW
  -1.0472,        # L_ELBOW
  -1.972222054,   # L_WRIST_ROLL
  -1.614429558,   # L_WRIST_PITCH
  -1.614429558,   # L_WRIST_YAW
  -3.0892,        # R_SHOULDER_PITCH
  -2.2515,        # R_SHOULDER_ROLL
  -2.618,         # R_SHOULDER_YAW
  -1.0472,        # R_ELBOW
  -1.972222054,   # R_WRIST_ROLL
  -1.614429558,   # R_WRIST_PITCH
  -1.614429558,    # R_WRIST_YAW
  -1.0471,    # Thumb 0
  -0.610865,  # Thumb 1
  0.0,        # Thumb 2
  0.0,        # Middle 0
  0.0,        # Middle 1
  0.0,        # Index 0
  0.0,        # Index 1
  -1.0471,    # Thumb 0
  -0.610865,  # Thumb 1
  0.0,        # Thumb 2
  0.0,        # Middle 0
  0.0,        # Middle 1
  0.0,        # Index 0
  0.0         # Index 1
]

JOINT_LIMITS_UPPER = [
  2.8798,        # L_LEG_HIP_PITCH
  2.9671,        # L_LEG_HIP_ROLL
  2.7576,        # L_LEG_HIP_YAW
  2.8798,        # L_LEG_KNEE
  0.5236,        # L_LEG_ANKLE_PITCH
  0.2618,        # L_LEG_ANKLE_ROLL
  2.8798,        # R_LEG_HIP_PITCH
  0.5236,        # R_LEG_HIP_ROLL
  2.7576,        # R_LEG_HIP_YAW
  2.8798,        # R_LEG_KNEE
  0.5236,        # R_LEG_ANKLE_PITCH
  0.2618,        # R_LEG_ANKLE_ROLL
  2.618,         # WAIST_YAW
  0.52,          # WAIST_ROLL
  0.52,          # WAIST_PITCH
  2.6704,        # L_SHOULDER_PITCH
  2.2515,        # L_SHOULDER_ROLL
  2.618,         # L_SHOULDER_YAW
  2.0944,        # L_ELBOW
  1.972222054,   # L_WRIST_ROLL
  1.614429558,   # L_WRIST_PITCH
  1.614429558,   # L_WRIST_YAW
  2.6704,        # R_SHOULDER_PITCH
  1.5882,        # R_SHOULDER_ROLL
  2.618,         # R_SHOULDER_YAW
  2.0944,        # R_ELBOW
  1.972222054,   # R_WRIST_ROLL
  1.614429558,   # R_WRIST_PITCH
  1.614429558,   # R_WRIST_YAW
  1.0471,     # Thumb 0
  1.0471,     # Thumb 1
  1.74532,    # Thumb 2
  1.5707,     # Middle 0
  1.74532,    # Middle 1
  1.5707,     # Index 0
  1.74532,     # Index 1
  1.0471,     # Thumb 0
  1.0471,     # Thumb 1
  1.74532,    # Thumb 2
  1.5707,     # Middle 0
  1.74532,    # Middle 1
  1.5707,     # Index 0
  1.74532     # Index 1
]
