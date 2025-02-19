# **AI-Powered Task Optimizer 🚀**  
A **Django-based AI-driven web application** that detects **mood through text and live facial expressions**. It features **real-time video analysis, a data visualization dashboard, and intelligent task suggestions** to enhance productivity and mental well-being.  

---

## **📌 Features**  

✅ **Text-Based Mood Classification** using **BERT + Cosine Similarity**  
✅ **Live Facial Emotion Detection** with **OpenCV (Face, Smile, and Eye Detection)**  
✅ **Real-Time Video Streaming** using Django's `StreamingHttpResponse`  
✅ **Interactive Data Analysis Dashboard** with **Plotly.js**  
✅ **Task Suggestions Based on Mood**  
✅ **Django Backend for Data Storage** (SQLite)  

---

## **📂 Project Structure**  

```
task_optimizer/
│── myapp/                      # Main Django App
│   ├── migrations/             # Database migrations
│   ├── __init__.py
│   ├── admin.py                # Django admin configuration
│   ├── apps.py                 # App configuration
│   ├── models.py               # Database Models (MoodEntry, etc.)
│   ├── tests.py                # Unit Tests
│   ├── urls.py                 # URL Routing
│   ├── views.py                # Main application logic (mood detection, video streaming, etc.)
│
│── static/                      # Static Files (CSS, JS, Images)
│   ├── analysis.css             # Styles for the data analysis dashboard
│   ├── detect_face.css          # Styles for the face detection UI
│   ├── styles.css               # General styles
│
│── task_optimizer/              # Django Project Settings
│   ├── __pycache__/             # Python cache files
│   ├── __init__.py
│   ├── asgi.py                  # ASGI Configuration
│   ├── settings.py              # Django Settings
│   ├── urls.py                  # Root URL Routing
│   ├── wsgi.py                  # WSGI Configuration
│
│── templates/                    # HTML Templates
│   ├── analyse_data.html         # Mood analysis dashboard
│   ├── detect_face.html          # Live facial emotion detection page
│   ├── detect_mood.html          # Text-based mood classification page
│   ├── home.html                 # Homepage
│   ├── suggest_task.html         # Task suggestions based on mood
│
│── db.sqlite3                     # SQLite Database
│── manage.py                      # Django Management Script
```

---

## **🚀 Installation & Setup**  

### **1️⃣ Clone the Repository**  
```sh
git clone https://github.com/your-username/task_optimizer.git
cd task_optimizer
```

### **2️⃣ Create a Virtual Environment**  
```sh
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate      # Windows
```

### **3️⃣ Install Dependencies**  
```sh
pip install -r requirements.txt
```

### **4️⃣ Run Migrations & Start Server**  
```sh
python manage.py migrate
python manage.py runserver
```

### **5️⃣ Access the Web App**  
Open **http://127.0.0.1:8000/** in your browser.

---

## **🖥️ Usage**  

### **🌟 Text-Based Mood Detection**  
1️⃣ Navigate to `Detect Mood`  
2️⃣ Enter a short text or answer psychological questions  
3️⃣ AI (BERT Model) detects mood based on semantic similarity  
4️⃣ The detected mood is stored and analyzed  

### **🎭 Real-Time Facial Emotion Recognition**  
1️⃣ Navigate to `Detect Face`  
2️⃣ The camera captures **facial expressions, eye activity, and brightness**  
3️⃣ Haar cascades (`face`, `smile`, `eye`) classify moods in **real-time**  
4️⃣ The result is displayed and logged  

### **📊 Mood Analysis Dashboard**  
1️⃣ Navigate to `Analyse Data`  
2️⃣ **Line Chart** 📈 - Tracks mood changes over time  
3️⃣ **Bar Chart** 📊 - Shows mood distribution  

### **💡 AI-Based Task Suggestions**  
1️⃣ Based on detected mood, relevant **tasks** are suggested  

---

## **📌 Key Technologies Used**  

✅ **Backend:** Django (Python)  
✅ **Frontend:** HTML, CSS, JavaScript  
✅ **Computer Vision:** OpenCV (Face, Smile, Eye Detection)  
✅ **Machine Learning:** BERT + Cosine Similarity (Text Analysis)  
✅ **Data Visualization:** Plotly.js  
✅ **Database:** SQLite  

---


---

## **🚀 Future Enhancements**  
🔹 **Sentiment Analysis with LLMs**  
🔹 **More Advanced Facial Expression Recognition**  
🔹 **Custom AI Models for Task Optimization**  

---

## **📌 Contributors**  
👤 **Param Parekh** - Developer  
📧 Email: parammparekh13@gmail.com 
