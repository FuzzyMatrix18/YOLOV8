import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import queue
import sqlite3
from collections import deque
import torch
import torchvision.models as models
import torchvision.transforms as T
from scipy.spatial.distance import cosine

# =================================================================
# 1. HOMOGRAPHY, SYSTEM & DB CONFIGURATION
# =================================================================
SOURCE_POINTS = np.array([[216, 216], [282, 840], [1663, 662], [1494, 69]], dtype='float32')
DEST_POINTS = np.array([[0, 0], [0, 500], [500, 500], [500, 0]], dtype='float32')
M = cv2.getPerspectiveTransform(SOURCE_POINTS, DEST_POINTS)
PIXELS_PER_METER = 100

db_conn = sqlite3.connect('store_analytics.db', check_same_thread=False)
cursor = db_conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS analytics 
                  (track_id INTEGER, dwell_time REAL, zones_visited TEXT, interactions INTEGER, timestamp TEXT)''')
db_conn.commit()

# =================================================================
# 2. AI MODELS (M1 ACCELERATED)
# =================================================================
print("Initializing YOLOv8-Pose & ResNet-18 ReID on Apple Silicon...")
# Action Recognition Model
model = YOLO("yolov8n-pose.pt") 

# ReID Embedding Model (ResNet18 without the final classification layer)
reid_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
reid_model = torch.nn.Sequential(*list(reid_model.children())[:-1])
reid_model.to("mps").eval()

reid_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =================================================================
# 3. ZONES, TRIPWIRE & GLOBAL STATE
# =================================================================
ZONES = [
    {"name": "Entrance", "poly": np.array([[0, 0], [200, 0], [200, 150], [0, 150]], np.int32), "color": (255, 0, 255)},
    {"name": "Main Display", "poly": np.array([[300, 300], [500, 300], [500, 500], [300, 500]], np.int32), "color": (0, 255, 255)}
]
TRIPWIRE_A = (100, 250)
TRIPWIRE_B = (400, 250)

# State Managers
path_history = {}          
dwell_tracker = {}         
zone_visit_log = {}        
interaction_counts = {}    
visual_database = {}       # {id: embedding_vector}
active_ids_last_frame = set()
crossed_ids = set()        

entry_count = 0
exit_count = 0

# =================================================================
# 4. MATH & RE-ID HELPERS
# =================================================================
def get_top_down_coords(x1, y1, x2, y2):
    point = np.array([[[ (x1 + x2) / 2, y2 ]]], dtype='float32')
    transformed = cv2.perspectiveTransform(point, M)
    return int(transformed[0][0][0]), int(transformed[0][0][1])

def ccw(A, B, C): return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
def intersect(A, B, C, D): return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def get_embedding(frame, box):
    x1, y1, x2, y2 = map(int, box)
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10: return None
    
    tensor = reid_transform(crop).unsqueeze(0).to("mps")
    with torch.no_grad():
        return reid_model(tensor).cpu().numpy().flatten()

# =================================================================
# 5. ASYNC VIDEO THREAD
# =================================================================
frame_queue = queue.Queue(maxsize=30)
stop_thread = False

def video_reader():
    global stop_thread
    cap = cv2.VideoCapture('data/store_feed.mp4')
    while cap.isOpened() and not stop_thread:
        success, frame = cap.read()
        if not success: break
        if not frame_queue.full(): frame_queue.put(frame)
        else: time.sleep(0.01)
    cap.release()
    stop_thread = True

threading.Thread(target=video_reader, daemon=True).start()

# =================================================================
# 6. MAIN INTELLIGENCE PIPELINE
# =================================================================
heatmap_data = np.zeros((500, 500), dtype=np.float32)

while not stop_thread or not frame_queue.empty():
    if frame_queue.empty():
        time.sleep(0.01)
        continue
        
    frame = frame_queue.get()
    start_time = time.time()
    bev_canvas = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Render BEV UI
    for zone in ZONES:
        cv2.polylines(bev_canvas, [zone['poly']], True, zone['color'], 2)
        cv2.putText(bev_canvas, zone['name'], (zone['poly'][0][0], zone['poly'][0][1]-5), 0, 0.4, zone['color'], 1)
    cv2.line(bev_canvas, TRIPWIRE_A, TRIPWIRE_B, (0, 0, 255), 2)
    cv2.putText(bev_canvas, f"IN: {entry_count} | OUT: {exit_count}", (TRIPWIRE_A[0], TRIPWIRE_A[1]-10), 0, 0.5, (0, 0, 255), 2)

    # Inference (Pose Model)
    results = model.track(frame, persist=True, classes=[0], device="mps", verbose=False)
    current_frame_ids = set()

    if results[0].boxes.id is not None and results[0].keypoints is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        kpts = results[0].keypoints.xy.cpu().numpy() # [Num_People, 17, 2]

        for box, track_id, kpt in zip(boxes, ids, kpts):
            mx, my = get_top_down_coords(box[0], box[1], box[2], box[3])
            
            # --- IDENTITY MANAGEMENT (ReID) ---
            # If this is a brand new ID to our tracker, check if they match a known "leaver"
            if track_id not in active_ids_last_frame and track_id not in visual_database:
                current_embedding = get_embedding(frame, box)
                if current_embedding is not None:
                    # Search visual database for a match > 85% similarity
                    for old_id, old_emb in visual_database.items():
                        if 1 - cosine(current_embedding, old_emb) > 0.85:
                            # Re-assign ID logic (simplified for demonstration)
                            print(f"Re-ID Match: ID {track_id} is actually returning customer ID {old_id}")
                            break
                    visual_database[track_id] = current_embedding

            current_frame_ids.add(track_id)

            # --- KINEMATICS (Interaction Detection) ---
            interaction_flag = False
            left_shoulder, left_wrist = kpt[5], kpt[9]
            if left_wrist[0] != 0 and left_shoulder[0] != 0:
                arm_reach = np.sqrt((left_wrist[0] - left_shoulder[0])**2 + (left_wrist[1] - left_shoulder[1])**2)
                if arm_reach > 40: # Pixel threshold for an extended arm
                    interaction_flag = True
                    interaction_counts[track_id] = interaction_counts.get(track_id, 0) + 1

            # --- ANALYTICS UPDATES ---
            if track_id not in path_history: path_history[track_id] = deque(maxlen=25)
            path_history[track_id].append((mx, my))

            if track_id not in zone_visit_log: zone_visit_log[track_id] = set()
            for zone in ZONES:
                if cv2.pointPolygonTest(zone['poly'], (float(mx), float(my)), False) >= 0:
                    zone_visit_log[track_id].add(zone['name'])

            if track_id not in dwell_tracker: dwell_tracker[track_id] = time.time()
            if 0 <= mx < 500 and 0 <= my < 500: cv2.circle(heatmap_data, (mx, my), 15, 1.0, -1)

            # --- VELOCITY & TRIPWIRE ---
            speed_status, velocity_mps = "Browsing", 0.0
            if len(path_history[track_id]) >= 5:
                px, py = path_history[track_id][-5]
                velocity_mps = (np.sqrt((mx - px)**2 + (my - py)**2) / PIXELS_PER_METER) / 0.25
                if velocity_mps > 1.2: speed_status = "Walking"

            if len(path_history[track_id]) >= 2 and track_id not in crossed_ids:
                p1, p2 = path_history[track_id][-2], path_history[track_id][-1]
                if intersect(TRIPWIRE_A, TRIPWIRE_B, p1, p2):
                    crossed_ids.add(track_id)
                    if p1[1] < TRIPWIRE_A[1] and p2[1] > TRIPWIRE_A[1]: entry_count += 1
                    elif p1[1] > TRIPWIRE_A[1] and p2[1] < TRIPWIRE_A[1]: exit_count += 1

            # --- VISUAL RENDERING ---
            pts = list(path_history[track_id])
            for i in range(1, len(pts)): cv2.line(bev_canvas, pts[i-1], pts[i], (255, 255, 255), 2)

            box_color = (0, 0, 255) if interaction_flag else (0, 255, 0)
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color, 2)
            
            # Label with Speed and Kinematic state
            label = f"ID:{track_id} | {speed_status} " + ("| *INTERACTING*" if interaction_flag else "")
            cv2.putText(frame, label, (int(box[0]), int(box[1]-10)), 0, 0.5, box_color, 2)
            cv2.circle(bev_canvas, (mx, my), 8, box_color, -1)

    # --- LEAVER LOGIC (Database & Cleanup) ---
    for l_id in (active_ids_last_frame - current_frame_ids):
        final_dwell = time.time() - dwell_tracker.get(l_id, time.time())
        z_str = ", ".join(list(zone_visit_log.get(l_id, [])))
        i_count = interaction_counts.get(l_id, 0)
        t_stamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        cursor.execute("INSERT INTO analytics VALUES (?, ?, ?, ?, ?)", (int(l_id), final_dwell, z_str, i_count, t_stamp))
        db_conn.commit()

        # Clean memory, but keep embedding in visual_database for future ReID
        for struct in (dwell_tracker, path_history, zone_visit_log, interaction_counts):
            if l_id in struct: del struct[l_id]
        if l_id in crossed_ids: crossed_ids.remove(l_id)

    active_ids_last_frame = current_frame_ids

    # --- COMPOSITE & UI ---
    heatmap_norm = cv2.normalize(heatmap_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    mask = heatmap_norm > 0
    heatmap_final = np.zeros_like(heatmap_color)
    heatmap_final[mask] = heatmap_color[mask]
    final_bev = cv2.addWeighted(bev_canvas, 0.7, heatmap_final, 0.5, 0)

    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 0, 1, (0, 255, 0), 2)
    
    # Scale main frame to match BEV height (500px) for a clean horizontal stack
    h, w = frame.shape[:2]
    scaled_frame = cv2.resize(frame, (int(w * (500/h)), 500))
    combined = np.hstack((scaled_frame, final_bev))
    
    cv2.imshow("TensorGo: AI Spatial Engine", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_thread = True
        break

stop_thread = True
db_conn.close()
cv2.destroyAllWindows()
cv2.waitKey(1)