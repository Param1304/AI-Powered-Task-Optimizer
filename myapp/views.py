from django.shortcuts import render
# import tensorflow as tf
# tf.get_logger().setLevel('ERROR')

# Create your views here.
def home(request):
    return render(request, 'home.html', {'mood':'Normal'})

import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from .models import MoodEntry

# Load the pretrained model
MODEL_PATH = r"C:\Users\Param\OneDrive\文档\Zidio_Project\saved_mental_status_bert"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = BertModel.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

# Reference sentences for mood classification
reference_sentences = {
    "Normal": "I feel okay today, nothing much to complain about.",
    "Normal":"I am motivated to work. No issues",
    "Normal":"No I am fine. I am okay",
    "Normal":"I am well to do",
    "Depression": "I feel like I am at the end, nothing I do is ever right.",
    "Depression": "I hardly find anything enjoyable or pleasurable",
    "Depression":"Everything is so bad",
    "Depression": "I am feeling Isolated and left over",
    "Suicidal": "I have given up on life. I wish everything would just end.",
    "Suicidal":"I have nothing to look onto. Everything is finished",
    "Suicidal":"I quit. I cannot continue anymore",
    "Anxiety": "I am really worried, I can't seem to relax.",
    "Anxiety":"I am sad and anxious. I am frustrated",
    "Anxiety":"Many time I canno control my thoughts. I do overthinking.",
    "Anxiety":"I cannot take decisions. I am feeling weak",
}

# Compute reference embeddings
reference_inputs = tokenizer(list(reference_sentences.values()), return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    reference_outputs = model(**reference_inputs)
reference_embeddings = reference_outputs.last_hidden_state[:, 0, :].numpy()

# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# View for detecting mood
def detect_mood(request):
    if request.method == "POST":
        responses = [request.POST[f"q{i}"] for i in range(1, 6)]

        # Process responses with BERT
        inputs = tokenizer(responses, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        sentence_embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        # Compare responses with reference embeddings
        predicted_labels = []
        for emb in sentence_embeddings:
            similarities = {label: cosine_similarity(emb, ref_emb) for label, ref_emb in zip(reference_sentences.keys(), reference_embeddings)}
            predicted_label = max(similarities, key=similarities.get)
            predicted_labels.append(predicted_label)

        # Determine the most common mood
        final_mood = max(set(predicted_labels), key=predicted_labels.count)
        MoodEntry.objects.create(text=" | ".join(responses), mood=final_mood)
        return render(request, "detect_mood.html", {"mood": final_mood})
    return render(request, "detect_mood.html")

def suggest_task(request,mood):
    tasks=[]
    if mood=="Normal":
        tasks=[
            "Start working on your most important project of the day.",
            "Take a short break, and then plan your day ahead.",
            "Meet with a colleague to discuss collaborative work."
        ]
    elif mood == "Depression":
        tasks = [
            "Take a walk outside to clear your mind.",
            "Try writing down your thoughts in a journal.",
            "Start with a small, achievable task like making a to-do list."
        ]
    elif mood == "Suicidal":
        tasks = [
            "Reach out to a mental health professional.",
            "Contact a friend or family member for support.",
            "Take a break and try to rest. Avoid overloading yourself."
        ]
    elif mood == "Anxiety":
        tasks = [
            "Practice deep breathing exercises for a few minutes.",
            "Organize your workspace to reduce stress.",
            "Take a break and listen to calming music."
        ]

    return render(request, "suggest_task.html", {"tasks": tasks, "mood": mood})

from datetime import datetime
import json
from django.db.models import Count 
def analyse_data(request):
    mood_entries = MoodEntry.objects.all().order_by('timestamp')
    timestamps = [entry.timestamp.strftime('%Y-%m-%d %H:%M') for entry in mood_entries]
    mood_mapping = {'Normal':0, 'Anxiety':1, 'Depression':2, 'Suicidal':3}
    moods = [mood_mapping[entry.mood] for entry in mood_entries]
    mood_counts = MoodEntry.objects.values('mood').annotate(count=Count('mood'))
    mood_categories = [item['mood'] for item in mood_counts]
    mood_counts = [item['count'] for item in mood_counts]
    context = {
        # 'timestamps': timestamps,
        # 'moods': moods,
        # 'mood_categories': mood_categories,
        # 'mood_counts': mood_counts
        'timestamps': json.dumps(timestamps),
        'moods': json.dumps(moods),
        'mood_categories': json.dumps(mood_categories),
        'mood_counts': json.dumps(mood_counts)
    }
    return render(request, 'analyse_data.html', context)

from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
import numpy as np

# Load OpenCV models for face, eyes, and smile detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


recent_emotions = []

# Emotion classification based on face brightness, smile, and eye detection
def classify_emotion(gray_face, face_region_color):
    global recent_emotions
    mean_intensity = np.mean(gray_face)
    smiles = smile_cascade.detectMultiScale(gray_face, scaleFactor=1.8, minNeighbors=25, minSize=(25, 25))
    eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.2, minNeighbors=15, minSize=(20, 20))
    # Decision rules 
    if len(smiles) > 0:
        emotion = "Happy (Normal)"
    elif len(eyes) == 0: 
        emotion = "Depression"
    elif mean_intensity > 100:
        emotion = "Normal"
    elif 130 < mean_intensity <= 180:
        emotion = "Anxiety"
    elif 80 < mean_intensity <= 130:
        emotion = "Depression"
    else:
        emotion = "Suicidal"

    recent_emotions.append(emotion)
    if len(recent_emotions) > 5:
        recent_emotions.pop(0)

    return max(set(recent_emotions), key=recent_emotions.count)  # Most frequent recent emotion

def generate_frames():
    cap = cv2.VideoCapture(0)  
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_region_gray = gray[y:y + h, x:x + w]
            face_region_color = frame[y:y + h, x:x + w]
            emotion = classify_emotion(face_region_gray, face_region_color)

            # Draw bounding box for face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Eye detection within the face region
            eyes = eye_cascade.detectMultiScale(face_region_gray, scaleFactor=1.2, minNeighbors=15, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_region_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

            # Smile detection within the face region
            smiles = smile_cascade.detectMultiScale(face_region_gray, scaleFactor=1.8, minNeighbors=25, minSize=(25, 25))
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(face_region_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# View for streaming video
def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

# View to render the HTML page
def detect_face(request):
    return render(request, 'detect_face.html')




# from django.shortcuts import render
# from django.http import StreamingHttpResponse
# import cv2
# import numpy as np

# # Load OpenCV face detection model
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# # Simple emotion classification based on heuristic (brightness of face)
# # def classify_emotion(gray_face):
# #     mean_intensity = np.mean(gray_face)
# #     if mean_intensity > 150:
# #         return "Normal"
# #     elif 100 < mean_intensity <= 150:
# #         return "Anxiety"
# #     elif 50 < mean_intensity <= 100:
# #         return "Depression"
# #     else:
# #         return "Suicidal"
# def classify_emotion(gray_face):
#     mean_intensity = np.mean(gray_face)
#     smiles = smile_cascade.detectMultiScale(gray_face, scaleFactor=1.8, minNeighbors=25, minSize=(25, 25))
#     eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.2, minNeighbors=15, minSize=(20, 20))
#     if len(smiles) > 0:
#         return "Happy (Normal)"
#     elif len(eyes) == 0:  # If no eyes detected, it might indicate closed eyes (fatigue/depression)
#         emotion = "Depression"
#     elif mean_intensity > 100:
#         return "Normal"
#     elif 150 < mean_intensity <= 200:
#         return "Anxiety"
#     elif 100 < mean_intensity <= 150:
#         return "Depression"
#     else:
#         return "Suicidal"
# # Video capture generator
# def generate_frames():
#     cap = cv2.VideoCapture(0)  # Open the webcam
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break

#         # Convert to grayscale for face detection
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

#         for (x, y, w, h) in faces:
#             face_region = gray[y:y + h, x:x + w]
#             emotion = classify_emotion(face_region)

#             # Draw bounding box and emotion label
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#         # Encode frame as JPEG
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()

#         yield (b'--frame\r\n'
#             b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#     cap.release()

# # View for streaming video
# def video_feed(request):
#     return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

# # View to render the HTML page
# def detect_face(request):
#     return render(request, 'detect_face.html')
