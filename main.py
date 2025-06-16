from fastapi import FastAPI, File, UploadFile
from fastapi import Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uvicorn

import cv2
import mediapipe as mp
import numpy as np

import firebase_admin
from firebase_admin import credentials, storage, firestore
from datetime import datetime

# get key content from enviroment variable
firebase_key_json = os.environ.get("FIREBASE_KEY_JSON")

# create file:
if not os.path.exists("firebase_key.json"):
    with open("firebase_key.json", "w") as f:
        f.write(firebase_key_json)

# this is for uploading the video to firebase
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
  "storageBucket": "runningformai-e5478.firebasestorage.app"
})
db = firestore.client()
bucket = storage.bucket()

app = FastAPI()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Use 8080 by default for local testing
    uvicorn.run("main:app", host="0.0.0.0", port=port)

# Allow CORS so your app can communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # when ready to deploy fix this to https://yourapp.com for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...), user_id: str = Form(...)):
    # Save video to disk
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the video and get feedback
    feedback = process_video(file_path)
    print(feedback)

    try:
      # Upload video to Firebase Storage
      blob = bucket.blob(f"videos/{user_id}/{file.filename}")
      blob.upload_from_filename(file_path)
      blob.make_public()  # You can remove this if you want access to be restricted

      videoURL = blob.public_url

      # Save feedback + video URL to Firestore
      db.collection("users").document(user_id).collection("videos").add({
        "videoURL": videoURL,
        "feedback": feedback,
        "timestamp": firestore.SERVER_TIMESTAMP
    })

      # Delete video after processing
      os.remove(file_path)

      return {
          "feedback": feedback,
          "videoURL": videoURL,
          "created_at": datetime.now()
      }
    except Exception as e:
        return {"error": str(e)}

   

def compute_angle(a, b, c):
    """Compute angle at point b given points a, b, c (in landmark format)."""
    ab = np.array([a.x - b.x, a.y - b.y])
    cb = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def process_video(file_path):
    frames_with_pose = 0 # counter to know if video does not contain someone running
    frame_inx = 0 # frame counter to only process every couple frames

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(file_path)

    left_knee_heights = []
    right_knee_heights = []
    left_wrist_ys = []
    right_wrist_ys = []
    nose_ys = []
    foot_strike_distances = []
    posture_angles = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_inx += 1 # increment and only process every 3 frames
        if frame_inx % 3 != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

             # Check how confident the detection is (use visibility or presence)
            visibility_scores = [lmk.visibility for lmk in lm]
            avg_visibility = np.mean(visibility_scores)

            # Only count as valid pose if visibility is high enough
            if avg_visibility < 0.5:
              continue  # skip this frame

            required_joints = [
              mp_pose.PoseLandmark.LEFT_HIP,
              mp_pose.PoseLandmark.RIGHT_HIP,
              mp_pose.PoseLandmark.LEFT_KNEE,
              mp_pose.PoseLandmark.RIGHT_KNEE,
              mp_pose.PoseLandmark.LEFT_ANKLE,
              mp_pose.PoseLandmark.RIGHT_ANKLE
            ]

            if any(lm[j].visibility < 0.5 for j in required_joints):
                continue  # skip low-confidence frame

            frames_with_pose += 1

            # Collect data for each frame
            # Normalize by shoulder-to-hip length to reduce scale effects
            torso_len = abs(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y - lm[mp_pose.PoseLandmark.LEFT_HIP].y)

            # Knee drive
            left_knee_heights.append(lm[mp_pose.PoseLandmark.LEFT_KNEE].y)
            right_knee_heights.append(lm[mp_pose.PoseLandmark.RIGHT_KNEE].y)

            # Arm swing
            left_wrist_ys.append(lm[mp_pose.PoseLandmark.LEFT_WRIST].y)
            right_wrist_ys.append(lm[mp_pose.PoseLandmark.RIGHT_WRIST].y)

            # Head stability
            nose_ys.append(lm[mp_pose.PoseLandmark.NOSE].y)

            # Foot landing (distance of foot from hip)
            left_foot_dist = abs(lm[mp_pose.PoseLandmark.LEFT_ANKLE].x - lm[mp_pose.PoseLandmark.LEFT_HIP].x)
            right_foot_dist = abs(lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x - lm[mp_pose.PoseLandmark.RIGHT_HIP].x)
            foot_strike_distances.append(max(left_foot_dist, right_foot_dist) / torso_len)

            # Posture (hip - shoulder - nose)
            angle = compute_angle(
                lm[mp_pose.PoseLandmark.LEFT_HIP],
                lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
                lm[mp_pose.PoseLandmark.NOSE]
            )
            posture_angles.append(angle)

    cap.release()
    pose.close()

    # ========== RULES & FEEDBACK ==========
    feedback = []

    # if pose didn't detect a person
    if (frames_with_pose < 10):
        feedback.append("No runner was detected in the video. Please try again with a clearer side-view video")
        return feedback


    # Knee Drive
    avg_knee_height = (np.mean(left_knee_heights) + np.mean(right_knee_heights)) / 2
    avg_hip_height = (lm[mp_pose.PoseLandmark.LEFT_HIP].y + lm[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
    if avg_knee_height > avg_hip_height:
        feedback.append("Try lifting your knees higher to improve stride power.")

    # Arm swing symmetry
    left_range = max(left_wrist_ys) - min(left_wrist_ys)
    right_range = max(right_wrist_ys) - min(right_wrist_ys)
    if abs(left_range - right_range) > 0.05:
        feedback.append("Your arm swing appears uneven. Try to swing both arms evenly.")

    # Head stability
    head_oscillation = max(nose_ys) - min(nose_ys)
    if head_oscillation > 0.1:
        feedback.append("Try to keep your head more stable while running.")

    # Overstriding
    avg_foot_strike = np.mean(foot_strike_distances)
    if avg_foot_strike > 0.25:
        feedback.append("Your foot is landing too far in front. Reduce overstriding for better efficiency.")

    # Posture
    avg_posture_angle = np.mean(posture_angles)
    if avg_posture_angle < 150 or avg_posture_angle > 170:
        feedback.append("Maintain a slight forward lean from the ankles without hunching.")

    if not feedback:
        feedback.append("Your running form looks balanced. Keep it up!")

    return feedback
