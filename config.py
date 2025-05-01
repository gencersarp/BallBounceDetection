# --- Real Environment Config ---
YOLO_MODEL_PATH_DEFAULT = 'yolov8n.pt' # Default model for real env
YOLO_BALL_CLASS = "sports ball"      # Class name for balls in YOLO model
YOLO_BALL_CONF_THRESHOLD = 0.3       # Min confidence for YOLO ball detection
OBSTACLE_CONF_THRESHOLD = 0.5       # Min confidence for detecting obstacles
YOLO_SCAN_INTERVAL = 5              # Run YOLO detection every N frames (and when no balls tracked)
HSV_TOLERANCE = (10, 40, 40)         # Tolerance (+/-) for H, S, V when creating dynamic range
MIN_CONTOUR_AREA_COLOR = 50          # Min area for a contour found during color tracking
# Classes to consider as obstacles
OBSTACLE_CLASSES = ["sports ball"]

# --- Simulated Environment Config ---
SIM_MIN_BALL_RADIUS = 5
SIM_MAX_BALL_RADIUS = 75
SIM_CIRCULARITY_MIN = 0.9
SIM_CIRCULARITY_MAX = 1.3

# --- General Config ---
# Tracking parameters
MAX_DISAPPEARED_FRAMES = 10 # Reduced slightly as color tracking might be less robust
MAX_DISTANCE_TRACKING = 90 # Increased slightly for associating YOLO detections

# Bounce detection parameters
BOUNCE_VELOCITY_THRESHOLD = 1
BOUNCE_MIN_HEIGHT_THRESHOLD = 2