import numpy as np
import argparse
import imutils
import cv2
import math
from collections import deque, defaultdict
from physics import calculate_distance, calculate_circularity, get_average_color_bgr
from config import *  # Import all configuration values from the new config file


# conditionally import YOLO
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[WARNING] ultralytics library not found. 'real' environment mode will be disabled.")
    # Define a dummy YOLO class if not available to avoid NameErrors later
    class YOLO:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return [] # Return empty list

# --- Configuration ---
# --- Environment Specific ---
# Set via command line argument --environment ['real', 'simulated']




class BallBounceCounter:
    def __init__(self, environment, video_path=None, yolo_model_path=YOLO_MODEL_PATH_DEFAULT):
        self.environment = environment if environment else 'real'  # Default to 'real' environment
        self.output_path = None  # Default output path is None

        # Parse optional output argument
        if args.get("output"):
            self.output_path = args["output"]

        # Ensure the output is saved in the 'outputs' folder
        if self.output_path:
            import os
            outputs_dir = os.path.join(os.getcwd(), 'outputs')
            if not os.path.exists(outputs_dir):
                os.makedirs(outputs_dir)
            self.output_path = os.path.join(outputs_dir, os.path.basename(self.output_path))
            print(f"[INFO] Output will be saved to: {self.output_path}")

        if self.environment == 'real' and not _YOLO_AVAILABLE:
            print("[ERROR] 'real' environment selected, but ultralytics library is not available.")
            print("[INFO] Switching to 'simulated' environment.")
            self.environment = 'simulated'

        self.args = {"video": video_path}
        self.yolo_model = None
        self.yolo_classes = {}
        self.yolo_ball_class_id = -1
        self.obstacles = [] # Store detected obstacles per frame

        print(f"[INFO] Environment set to: {self.environment}")

        # Initialize video stream
        # (Same as before)
        if not self.args.get("video", False):
            print("[INFO] Starting video stream from webcam...")
            self.camera = cv2.VideoCapture(0)
        else:
            print(f"[INFO] Opening video file: {self.args['video']}...")
            self.camera = cv2.VideoCapture(self.args["video"])
        if not self.camera.isOpened():
            print("[ERROR] Cannot open video source.")
            exit()

        # Load YOLO model only for 'real' environment
        if self.environment == 'real':
            print(f"[INFO] Loading YOLO model for real environment from {yolo_model_path}...")
            try:
                self.yolo_model = YOLO(yolo_model_path)
                self.yolo_classes = self.yolo_model.model.names
                # Find the index for the ball class
                try:
                    # Get class names as a list
                    class_names_list = list(self.yolo_classes.values())
                    self.yolo_ball_class_id = class_names_list.index(YOLO_BALL_CLASS)
                    print(f"[INFO] YOLO model loaded. Ball class '{YOLO_BALL_CLASS}' has ID {self.yolo_ball_class_id}.")
                except ValueError:
                    print(f"[ERROR] YOLO model does not contain the specified ball class: '{YOLO_BALL_CLASS}'")
                    print(f"[INFO] Available classes: {self.yolo_classes}")
                    self.yolo_model = None # Disable YOLO if ball class not found
            except Exception as e:
                print(f"[ERROR] Failed to load YOLO model: {e}")
                self.yolo_model = None # Disable YOLO on error
        else:
             print("[INFO] YOLO model not loaded for simulated environment.")


        # Tracking variables
        self.next_object_id = 0
        # Stores {objectID: {"centroid": (x, y), "hsv_lower": (h,s,v), "hsv_upper": (h,s,v), "display_color": (B,G,R), "history": deque, "bounces": 0, "disappeared_frames": 0, "is_bouncing": False, "peak_y_after_bounce": float('inf')}}
        self.tracked_objects = {}
        self.frame_height = None
        self.frame_width = None
        self.lost_tracks = {}  # {id: {'color': ..., 'centroid': ..., 'bounces': ..., 'lost_frame': ...}}
        self.lost_track_ttl = 30  # frames to keep lost tracks for reassociation
        self.frame_count = 0

        # Initialize video writer if output path is specified
        self.video_writer = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, 30, (800, 600))  # Assuming 800x600 resolution

    def _calculate_dynamic_hsv_range(self, frame, bbox):
        """Calculates dynamic HSV range based on the center region of a bounding box."""
        x1, y1, x2, y2 = bbox
        # Ensure coordinates are within frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.frame_width, x2), min(self.frame_height, y2)
        if x1 >= x2 or y1 >= y2: return None, None # Invalid bbox

        # Define a smaller central ROI within the bbox to avoid edges
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        roi_w, roi_h = (x2 - x1) // 3, (y2 - y1) // 3 # Use inner third
        roi_x1, roi_y1 = center_x - roi_w // 2, center_y - roi_h // 2
        roi_x2, roi_y2 = center_x + roi_w // 2, center_y + roi_h // 2
        roi_x1, roi_y1 = max(x1, roi_x1), max(y1, roi_y1)
        roi_x2, roi_y2 = min(x2, roi_x2), min(y2, roi_y2)

        if roi_x1 >= roi_x2 or roi_y1 >= roi_y2: # If inner ROI is invalid, use full bbox
             roi = frame[y1:y2, x1:x2]
        else:
             roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi.size == 0: return None, None

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Calculate median HSV values (more robust to outliers than mean)
        # Ignore pixels with very low saturation or value (can distort hue)
        mask = (hsv_roi[:, :, 1] > 30) & (hsv_roi[:, :, 2] > 40)
        if np.count_nonzero(mask) < 10: # Not enough valid pixels, fallback?
             # Fallback: use mean of the whole ROI
             h = int(np.mean(hsv_roi[:,:,0]))
             s = int(np.mean(hsv_roi[:,:,1]))
             v = int(np.mean(hsv_roi[:,:,2]))
             if np.isnan(h) or np.isnan(s) or np.isnan(v): return None, None # Check for NaN
        else:
             h = int(np.median(hsv_roi[:, :, 0][mask]))
             s = int(np.median(hsv_roi[:, :, 1][mask]))
             v = int(np.median(hsv_roi[:, :, 2][mask]))


        h_tol, s_tol, v_tol = HSV_TOLERANCE

        # Calculate lower and upper bounds, handling Hue wrap-around
        lower_h = (h - h_tol) % 180
        upper_h = (h + h_tol) % 180
        lower_s = max(0, s - s_tol)
        upper_s = min(255, s + s_tol)
        lower_v = max(0, v - v_tol)
        upper_v = min(255, v + v_tol)

        lower_bound = np.array([lower_h, lower_s, lower_v])
        upper_bound = np.array([upper_h, upper_s, upper_v])

        # Store bounds that handle hue wrap-around if necessary
        # For simplicity now, we assume no wrap-around needed for the range itself
        # A more robust implementation would return two ranges if h_tol causes wrap
        if lower_h > upper_h: # Wrap around case
             # For tracking, we'll need to handle this (e.g., two masks)
             # For now, just return the calculated bounds, handle in tracking
             pass

        # print(f"  [HSV Range Calc] Center HSV: ({h},{s},{v}), Range: {lower_bound} -> {upper_bound}") # Debug
        return lower_bound, upper_bound


    def _register_object(self, centroid, hsv_lower, hsv_upper, display_color, history_len=30):
        """Registers a new object with its dynamic HSV range."""
        new_id = self.next_object_id
        self.tracked_objects[new_id] = {
            "centroid": centroid,
            "hsv_lower": hsv_lower,
            "hsv_upper": hsv_upper,
            "display_color": display_color, # Store BGR for display
            "history": deque(maxlen=history_len),
            "bounces": 0,
            "disappeared_frames": 0,
            "is_bouncing": False,
            "peak_y_after_bounce": float('inf')
        }
        self.tracked_objects[new_id]["history"].append(centroid)
        print(f"[INFO] Registered New Ball ID: {new_id} at {centroid} with HSV range {hsv_lower} -> {hsv_upper}")
        self.next_object_id += 1
        return new_id

    def _run_yolo_scan(self, frame):
        """Runs YOLO detection for both balls and obstacles."""
        if self.yolo_model is None: return [], []

        yolo_ball_detections = [] # List of {'box': (x1,y1,x2,y2), 'centroid': (cx,cy)}
        detected_obstacles = []   # List of {'box': (x1,y1,x2,y2), 'class': name}

        results = self.yolo_model(frame, verbose=False, conf=min(YOLO_BALL_CONF_THRESHOLD, OBSTACLE_CONF_THRESHOLD))

        if results and results[0].boxes:
            for box in results[0].boxes:
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Check for Ball
                if cls_id == self.yolo_ball_class_id and conf >= YOLO_BALL_CONF_THRESHOLD:
                    yolo_ball_detections.append({'box': (x1, y1, x2, y2), 'centroid': (cx, cy)})
                # Check for Obstacle
                elif cls_id in self.yolo_classes and self.yolo_classes[cls_id] in OBSTACLE_CLASSES and conf >= OBSTACLE_CONF_THRESHOLD:
                     detected_obstacles.append({"box": (x1, y1, x2, y2), "class": self.yolo_classes[cls_id]})

        return yolo_ball_detections, detected_obstacles

    def _update_tracks_with_yolo(self, frame, yolo_ball_detections):
        """Associates YOLO detections with existing tracks and registers new ones."""
        if not yolo_ball_detections: # No balls detected by YOLO
             # Mark all current tracks as potentially disappeared if no YOLO detections
             for object_id in self.tracked_objects.keys():
                  self.tracked_objects[object_id]["disappeared_frames"] += 1 # Increment disappearance
             return

        yolo_centroids = np.array([d['centroid'] for d in yolo_ball_detections])
        yolo_boxes = [d['box'] for d in yolo_ball_detections]

        if not self.tracked_objects: # No objects currently tracked, register all YOLO detections
            for i, box in enumerate(yolo_boxes):
                centroid = yolo_centroids[i]
                hsv_lower, hsv_upper = self._calculate_dynamic_hsv_range(frame, box)
                if hsv_lower is not None:
                    # Get avg BGR color for display from the contour within the box
                    # For simplicity, just use a fixed color or derive from HSV center
                    display_color = cv2.cvtColor(np.uint8([[hsv_lower]]), cv2.COLOR_HSV2BGR)[0][0].tolist() # Approx color
                    self._register_object(tuple(centroid), hsv_lower, hsv_upper, tuple(display_color))
            return

        # --- Association Logic ---
        object_ids = list(self.tracked_objects.keys())
        previous_centroids = np.array([d["centroid"] for d in self.tracked_objects.values()])

        # Calculate distance matrix (tracked_rows, yolo_cols)
        D = np.array([[calculate_distance(prev, yolo) for yolo in yolo_centroids] for prev in previous_centroids])

        # Greedy association
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols: continue

            if D[row, col] < MAX_DISTANCE_TRACKING: # Matched!
                object_id = object_ids[row]
                yolo_centroid = tuple(yolo_centroids[col])
                # print(f"  [YOLO Update] Matched Track ID {object_id} with YOLO detection at {yolo_centroid}") # Debug
                # Correct the centroid based on YOLO detection
                self.tracked_objects[object_id]["centroid"] = yolo_centroid
                self.tracked_objects[object_id]["history"].append(yolo_centroid) # Add corrected pos to history
                self.tracked_objects[object_id]["disappeared_frames"] = 0 # Reset disappearance

                # Optional: Recalculate HSV range based on the new box?
                hsv_lower, hsv_upper = self._calculate_dynamic_hsv_range(frame, yolo_boxes[col])
                if hsv_lower is not None:
                   self.tracked_objects[object_id]["hsv_lower"] = hsv_lower
                   self.tracked_objects[object_id]["hsv_upper"] = hsv_upper

                used_rows.add(row)
                used_cols.add(col)

        # Handle Unmatched YOLO Detections (Register as New)
        unused_cols = set(range(len(yolo_ball_detections))).difference(used_cols)
        for col in unused_cols:
            box = yolo_boxes[col]
            centroid = yolo_centroids[col]
            hsv_lower, hsv_upper = self._calculate_dynamic_hsv_range(frame, box)
            if hsv_lower is not None:
                 # Get approx BGR color for display
                 # Find contour within box to get better display color? Too slow maybe.
                 # Use HSV center converted to BGR
                 center_h = (hsv_lower[0] + hsv_upper[0]) // 2 # Approx center Hue
                 center_s = (hsv_lower[1] + hsv_upper[1]) // 2
                 center_v = (hsv_lower[2] + hsv_upper[2]) // 2
                 display_color = cv2.cvtColor(np.uint8([[[center_h, center_s, center_v]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
                 self._register_object(tuple(centroid), hsv_lower, hsv_upper, tuple(display_color))


        # Handle Unmatched Tracks (Increment disappeared frames)
        unused_rows = set(range(len(object_ids))).difference(used_rows)
        for row in unused_rows:
            object_id = object_ids[row]
            self.tracked_objects[object_id]["disappeared_frames"] += 1
            # print(f"  [YOLO Update] Track ID {object_id} not matched. Disappeared: {self.tracked_objects[object_id]['disappeared_frames']}") # Debug
            # Deregistration happens in the color tracking part


    def _track_balls_by_color(self, frame, hsv_frame):
        """Tracks balls frame-to-frame using their dynamic HSV ranges."""
        ids_to_deregister = []
        for object_id, data in self.tracked_objects.items():
            hsv_lower = data["hsv_lower"]
            hsv_upper = data["hsv_upper"]
            last_centroid = data["centroid"]

            # Skip processing if HSV bounds are None
            if hsv_lower is None or hsv_upper is None:
                print(f"[WARNING] Skipping object ID {object_id} due to invalid HSV bounds.")
                continue

            # --- Create Mask using HSV Range ---
            # Handle Hue wrap-around
            if hsv_lower[0] > hsv_upper[0]: # Hue wraps around 180
                lower_bound1 = hsv_lower
                upper_bound1 = np.array([179, hsv_upper[1], hsv_upper[2]])
                lower_bound2 = np.array([0, hsv_lower[1], hsv_lower[2]])
                upper_bound2 = hsv_upper
                mask1 = cv2.inRange(hsv_frame, lower_bound1, upper_bound1)
                mask2 = cv2.inRange(hsv_frame, lower_bound2, upper_bound2)
                mask = cv2.bitwise_or(mask1, mask2)
            else: # Normal case
                mask = cv2.inRange(hsv_frame, hsv_lower, hsv_upper)

            # Morphology
            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=2)

            # Find contours in the mask
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            best_match_contour = None
            min_dist = MAX_DISTANCE_TRACKING # Use tracking distance as threshold

            if cnts:
                # Find the contour closest to the last known position
                for c in cnts:
                    if cv2.contourArea(c) < MIN_CONTOUR_AREA_COLOR: continue # Filter small contours

                    M = cv2.moments(c)
                    if M["m00"] <= 0: continue
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    dist = calculate_distance(last_centroid, center)

                    if dist < min_dist:
                        min_dist = dist
                        best_match_contour = c

            # Update track if a good match was found
            if best_match_contour is not None:
                M = cv2.moments(best_match_contour)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                self.tracked_objects[object_id]["centroid"] = center
                self.tracked_objects[object_id]["history"].append(center)
                self.tracked_objects[object_id]["disappeared_frames"] = 0
            else: # No matching contour found
                self.tracked_objects[object_id]["disappeared_frames"] += 1
                if self.tracked_objects[object_id]["disappeared_frames"] > MAX_DISAPPEARED_FRAMES:
                    ids_to_deregister.append(object_id)

        # Deregister objects that have disappeared
        for object_id in ids_to_deregister:
            if object_id in self.tracked_objects:
                 print(f"[INFO] Deregistering object ID {object_id} (lost track).")
                 del self.tracked_objects[object_id]


    # --- Simulated Environment Detection ---
    def _detect_balls_simulated(self, frame):
        """Detects balls in simulated environment using Hough Circle Transform."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                   param1=50, param2=30, minRadius=SIM_MIN_BALL_RADIUS, maxRadius=SIM_MAX_BALL_RADIUS)

        detected_centroids_with_colors = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            valid_circles = []

            # Check for overlapping circles
            for (x, y, radius) in circles:
                is_inside_another = False
                for (vx, vy, vr) in valid_circles:
                    distance = math.sqrt((x - vx) ** 2 + (y - vy) ** 2)
                    if distance + radius <= vr:  # Current circle is inside another
                        is_inside_another = True
                        break
                if not is_inside_another:
                    valid_circles.append((x, y, radius))

            for (x, y, radius) in valid_circles:
                # Calculate circularity (optional, can be removed if not needed)
                circ = 1.0  # Hough circles are inherently circular
                if SIM_CIRCULARITY_MIN < circ < SIM_CIRCULARITY_MAX:
                    # Get BGR color for display/tracking association
                    display_color = frame[y, x].tolist()  # Approximate color at the circle center
                    detected_centroids_with_colors.append(((x, y), display_color))

        return detected_centroids_with_colors, []

    # --- Simulation Environment Tracker Update (uses BGR color for association) ---
    def _update_tracker_simulated(self, detected_centroids_with_colors):
        """Updates tracker for simulated env based on centroid and BGR color."""
        # This is similar to the previous _update_tracker, but associates based
        # on distance and potentially color similarity if needed.
        # For simplicity, let's stick to distance-based association for simulation.

        # Handle case of no detections
        if not detected_centroids_with_colors:
            ids_to_deregister = []
            for object_id in self.tracked_objects.keys():
                self.tracked_objects[object_id]["disappeared_frames"] += 1
                if self.tracked_objects[object_id]["disappeared_frames"] > MAX_DISAPPEARED_FRAMES:
                    ids_to_deregister.append(object_id)
            for object_id in ids_to_deregister:
                 if object_id in self.tracked_objects:
                     print(f"[INFO] Deregistering object ID {object_id} (disappeared).")
                     del self.tracked_objects[object_id]
            return

        input_centroids = np.array([c for c, _ in detected_centroids_with_colors])
        input_colors = [color for _, color in detected_centroids_with_colors] # BGR colors

        # Handle case of no currently tracked objects
        if not self.tracked_objects:
            for i, centroid in enumerate(input_centroids):
                # In simulation, store BGR color as 'display_color', no HSV range needed
                self._register_object_simulated(tuple(centroid), input_colors[i])
            return

        # Association Logic (Distance Based)
        object_ids = list(self.tracked_objects.keys())
        previous_centroids = np.array([d["centroid"] for d in self.tracked_objects.values()])
        D = np.array([[calculate_distance(prev, new) for new in input_centroids] for prev in previous_centroids])
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols: continue
            if D[row, col] < MAX_DISTANCE_TRACKING:
                object_id = object_ids[row]
                self.tracked_objects[object_id]["centroid"] = tuple(input_centroids[col])
                self.tracked_objects[object_id]["history"].append(tuple(input_centroids[col]))
                self.tracked_objects[object_id]["disappeared_frames"] = 0
                # Optionally update display color
                # self.tracked_objects[object_id]["display_color"] = input_colors[col]
                used_rows.add(row)
                used_cols.add(col)

        # Handle Unmatched
        unused_rows = set(range(len(object_ids))).difference(used_rows)
        unused_cols = set(range(len(input_centroids))).difference(used_cols)
        ids_to_deregister = []
        for row in unused_rows:
            object_id = object_ids[row]
            self.tracked_objects[object_id]["disappeared_frames"] += 1
            if self.tracked_objects[object_id]["disappeared_frames"] > MAX_DISAPPEARED_FRAMES:
                 ids_to_deregister.append(object_id)
        for object_id in ids_to_deregister:
             if object_id in self.tracked_objects:
                 print(f"[INFO] Deregistering object ID {object_id} (unmatched/disappeared).")
                 del self.tracked_objects[object_id]
        for col in unused_cols:
            self._register_object_simulated(tuple(input_centroids[col]), input_colors[col])

        # Adjust color similarity threshold and ensure re-association logic is robust
        def _is_color_similar(color1, color2, threshold=500):
            return np.linalg.norm(np.array(color1) - np.array(color2)) < threshold
    
        # Handle lost balls and re-associate based on color similarity
        for object_id in ids_to_deregister:
            if object_id in self.tracked_objects:
                lost_ball_data = self.tracked_objects[object_id]
                lost_color = lost_ball_data["display_color"]
                lost_bounces = lost_ball_data["bounces"]

                for col in unused_cols:
                    detected_color = input_colors[col]
                    if _is_color_similar(lost_color, detected_color):
                        # Re-associate the lost ball with the new detection
                        self.tracked_objects[object_id]["centroid"] = tuple(input_centroids[col])
                        self.tracked_objects[object_id]["history"].append(tuple(input_centroids[col]))
                        self.tracked_objects[object_id]["disappeared_frames"] = 0
                        self.tracked_objects[object_id]["bounces"] = lost_bounces
                        used_cols.add(col)
                        unused_cols.remove(col)  # Ensure this detection is not reused
                        print(f"[INFO] Re-associated lost ball ID {object_id} with new detection.")
                        break

    def _register_object_simulated(self, centroid, display_color, history_len=30):
        """Registers object for simulated env (stores BGR color)."""
        # Try reassociation first
        if self._try_reassociate(centroid, display_color):
            return
        new_id = self.next_object_id
        self.tracked_objects[new_id] = {
            "centroid": centroid,
            "hsv_lower": None, # Not used
            "hsv_upper": None, # Not used
            "display_color": display_color, # Store BGR for display
            "history": deque(maxlen=history_len),
            "bounces": 0,
            "disappeared_frames": 0,
            "is_bouncing": False,
            "peak_y_after_bounce": float('inf')
        }
        self.tracked_objects[new_id]["history"].append(centroid)
        print(f"[INFO] Registered New Simulated Object ID: {new_id} at {centroid} with color {display_color}")
        self.next_object_id += 1
        return new_id

    def _is_color_similar(self, c1, c2, threshold=50):
        if c1 is None or c2 is None:
            return False
        return np.linalg.norm(np.array(c1) - np.array(c2)) < threshold

    def _cleanup_lost_tracks(self):
        expired = [k for k, v in self.lost_tracks.items() if self.frame_count - v['lost_frame'] > self.lost_track_ttl]
        for k in expired:
            del self.lost_tracks[k]

    def _try_reassociate(self, new_centroid, new_color):
        for lost_id, lost_data in self.lost_tracks.items():
            if self._is_color_similar(lost_data['color'], new_color) and \
               np.linalg.norm(np.array(lost_data['centroid']) - np.array(new_centroid)) < MAX_DISTANCE_TRACKING * 2:
                # Reassociate
                self.tracked_objects[lost_id] = {
                    'centroid': new_centroid,
                    'hsv_lower': None,
                    'hsv_upper': None,
                    'display_color': new_color,
                    'history': deque([new_centroid], maxlen=30),
                    'bounces': lost_data['bounces'],
                    'disappeared_frames': 0,
                    'is_bouncing': False,
                    'peak_y_after_bounce': float('inf'),
                }
                del self.lost_tracks[lost_id]
                print(f"[INFO] Re-associated lost ball ID {lost_id} with new detection.")
                return True
        return False

    # --- Bounce Checking ---
    def _check_bounce(self, object_id, history, current_obstacles):
        """Checks if a ball bounced, avoiding obstacle collisions (obstacles only relevant in 'real' mode)."""
        # (Code is the same as previous version)
        if len(history) < 3: return False
        p0, p1, p2 = history[-3], history[-2], history[-1]
        if None in [p0, p1, p2]: return False
        vy1 = p1[1] - p0[1]
        vy2 = p2[1] - p1[1]
        current_y = p2[1]
        # Ensure object_id still exists before accessing
        if object_id not in self.tracked_objects: return False
        obj_data = self.tracked_objects[object_id]

        # Modify bounce detection to consider only bounces in the lower part of the frame
        if self.environment == 'real':
            frame_lower_bound = int(self.frame_height * 0.68)  # Lower 25% for real mode
        elif self.environment == 'simulated':
            frame_lower_bound = int(self.frame_height * 0.90)  # Lower 10% for simulated mode
        else:
            return False  # Unknown environment, ignore bounce

        if current_y < frame_lower_bound:
            return False  # Ignore bounces outside the specified lower part of the frame

        vel_change_indicates_bounce = (vy1 > BOUNCE_VELOCITY_THRESHOLD and vy2 < -BOUNCE_VELOCITY_THRESHOLD)
        is_already_bouncing = obj_data["is_bouncing"]
        collision_with_obstacle = False
        if self.environment == 'real' and vel_change_indicates_bounce:
            ball_radius_at_bounce = 10
            ball_bbox = (p2[0] - ball_radius_at_bounce, p2[1] - ball_radius_at_bounce,
                         p2[0] + ball_radius_at_bounce, p2[1] + ball_radius_at_bounce)
            for obs in current_obstacles:
                ox1, oy1, ox2, oy2 = obs["box"]
                if not (ball_bbox[2] < ox1 or ball_bbox[0] > ox2 or ball_bbox[3] < oy1 or ball_bbox[1] > oy2):
                    collision_with_obstacle = True
                    break
        if vel_change_indicates_bounce and not is_already_bouncing and not collision_with_obstacle:
            obj_data["bounces"] += 1
            obj_data["is_bouncing"] = True
            obj_data["peak_y_after_bounce"] = current_y
            print(f"[BOUNCE] Ball ID {object_id} bounced! Count: {obj_data['bounces']}")
            return True
        if obj_data["is_bouncing"]:
             obj_data["peak_y_after_bounce"] = min(obj_data["peak_y_after_bounce"], current_y)
             bounce_start_y = p1[1]
             min_height_reached = (current_y < bounce_start_y - BOUNCE_MIN_HEIGHT_THRESHOLD)
             if vy2 > BOUNCE_VELOCITY_THRESHOLD or min_height_reached:
                 obj_data["is_bouncing"] = False
                 obj_data["peak_y_after_bounce"] = float('inf')
        return False

    def _write_frame(self, frame):
        if self.video_writer:
            self.video_writer.write(frame)

    def run(self):
        """Main processing loop."""
        frame_count = 0
        while True:
            grabbed, frame = self.camera.read()
            self.frame_count += 1
            frame_count += 1
            if not grabbed: break

            frame = imutils.resize(frame, width=800)
            if self.frame_height is None:
                self.frame_height, self.frame_width = frame.shape[:2]
                print(f"[INFO] Frame dimensions: {self.frame_width}x{self.frame_height}")

            # --- Environment Specific Processing ---
            if self.environment == 'real':
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # Periodic YOLO Scan and Update
                if frame_count % YOLO_SCAN_INTERVAL == 0 or not self.tracked_objects:
                    # print(f"[INFO] Frame {frame_count}: Running YOLO Scan...") # Debug
                    yolo_balls, detected_obstacles = self._run_yolo_scan(frame)
                    self.obstacles = detected_obstacles # Update obstacle list
                    self._update_tracks_with_yolo(frame, yolo_balls)
                # Frame-to-frame Color Tracking
                self._track_balls_by_color(frame, hsv_frame)

            elif self.environment == 'simulated':
                # Simple detection and tracking for simulated env
                detected_centroids_with_colors, _ = self._detect_balls_simulated(frame)
                self._update_tracker_simulated(detected_centroids_with_colors) # Use the simulation tracker
                self.obstacles = [] # No obstacles in simulated mode

            else:
                 print(f"[ERROR] Unknown environment: {self.environment}")
                 break

            # --- Visualization ---
            # Draw obstacles (only in real mode)
            if self.environment == 'real':
                for obs in self.obstacles:
                    x1, y1, x2, y2 = obs["box"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue
                    cv2.putText(frame, obs["class"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw tracked objects and check for bounces
            # Iterate over a copy of keys for safe deletion during iteration
            current_object_ids = list(self.tracked_objects.keys())
            for object_id in current_object_ids:
                 # Re-check existence in case it was deregistered by color tracker
                 if object_id not in self.tracked_objects: continue

                 data = self.tracked_objects[object_id]
                 centroid = data["centroid"]
                 history = data["history"]
                 display_color = data["display_color"] # Use stored BGR color
                 bounces = data["bounces"]

                 # Draw centroid and text
                 cv2.circle(frame, centroid, 6, display_color, -1)
                 # White background for text
                 cv2.putText(frame, f"ID {object_id}", (centroid[0] - 15, centroid[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                 # Color text foreground
                 cv2.putText(frame, f"ID {object_id}", (centroid[0] - 15, centroid[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 1)
                 # Bounce count text
                 cv2.putText(frame, f"B: {bounces}", (centroid[0] - 15, centroid[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3) # Black outline
                 cv2.putText(frame, f"B: {bounces}", (centroid[0] - 15, centroid[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # Green text

                 # Draw trail
                 for i in range(1, len(history)):
                     if history[i - 1] is None or history[i] is None: continue
                     thickness = max(1, int(np.sqrt(len(history) / float(i + 1)) * 2.5))
                     cv2.line(frame, history[i - 1], history[i], display_color, thickness)

                 # Check for bounces, passing current obstacles
                 self._check_bounce(object_id, history, self.obstacles)


            # Display Output Frame
            cv2.imshow("Ball Bounce Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break

            # Write the processed frame to the output video
            self._write_frame(frame)

        # Release video writer if used
        if self.video_writer:
            self.video_writer.release()

        # --- Cleanup ---
        # (Same as before)
        print("[INFO] Cleaning up...")
        self.camera.release()
        cv2.destroyAllWindows()
        # Output counts for lost balls as well
        print("--------------------")
        print("Final Bounce Counts:")
        final_counts = sorted(self.tracked_objects.items())
        if final_counts:
            for obj_id, data in final_counts:
                print(f"  Ball ID {obj_id}: {data['bounces']} bounces")
        else:
            print("  No balls tracked.")
        print("--------------------")


# --- Script Execution ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Real-time ball bounce counter, environment-aware with dynamic color tracking.")
    ap.add_argument("-e", "--environment", required=True, choices=['real', 'simulated'],
                        help="Specify the operating environment ('real' or 'simulated').")
    ap.add_argument("-v", "--video", help="Path to the (optional) video file.")
    ap.add_argument("-m", "--model", default=YOLO_MODEL_PATH_DEFAULT,
                        help=f"Path to the YOLOv8 model file (used in 'real' environment). Default: {YOLO_MODEL_PATH_DEFAULT}")
    ap.add_argument("-o", "--output", help="Path to save the output video file.")
    args = vars(ap.parse_args())

    counter = BallBounceCounter(environment=args["environment"],
                                video_path=args["video"],
                                yolo_model_path=args["model"])
    counter.run()