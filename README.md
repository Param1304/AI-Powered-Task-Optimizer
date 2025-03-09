# **AI-Powered Task Optimizer        **   
A **Django-based AI-driven web application** that detects **mood through text and live facial 
expressions**. It features **real-time video analysis, a data visualization dashboard, and intelligent 
task suggestions** to enhance productivity and mental well-being.   
 --- 
 
## **     Features**   
 
   **Text-Based Mood Classification** using **BERT + Cosine Similarity**   
   **Live Facial Emotion Detection** with **OpenCV (Face, Smile, and Eye Detection)**   
   **Real-Time Video Streaming** using Django's `StreamingHttpResponse`   
   **Interactive Data Analysis Dashboard** with **Plotly.js**   
   **Task Suggestions Based on Mood**   
   **Django Backend for Data Storage** (SQLite)   
 --- 
 
## **   Project Structure**   
 
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
│── static/                     
│   
│   
├── analysis.css             
 # Static Files (CSS, JS, Images) 
# Styles for the data analysis dashboard 
├── detect_face.css          
│   
│ 
├── styles.css               
│── task_optimizer/             
│   
│   
│   
│   
│   
│   
│ 
├── __pycache__/             
├── __init__.py 
├── asgi.py                 
├── settings.py              
├── urls.py                  
├── wsgi.py                 
│── templates/                    
│   
│   
│   
│   
│   
│ 
├── analyse_data.html         
├── detect_face.html          
├── detect_mood.html         
├── home.html                
├── suggest_task.html         
│── db.sqlite3                     
│── manage.py                     
``` --- 
## **        
# Styles for the face detection UI 
# General styles 
 # Django Project Settings 
# Python cache files 
 # ASGI Configuration 
# Django Settings 
# Root URL Routing 
 # WSGI Configuration 
# HTML Templates 
# Mood analysis dashboard 
# Live facial emotion detection page 
 # Text-based mood classification page 
 # Homepage 
# Task suggestions based on mood 
# SQLite Database 
 # Django Management Script 
Installation & Setup**   
### **1️ Clone the Repository**   
```sh 
git clone https://github.com/your-username/task_optimizer.git 
cd task_optimizer 
``` 
### **2️ Create a Virtual Environment**   
```sh 
python -m venv venv 
source venv/bin/activate  # Mac/Linux 
venv\Scripts\activate      
``` 
# Windows 
### **3 Install Dependencies**   
```sh 
pip install -r requirements.txt 
``` 
### **4️ Run Migrations & Start Server**   
```sh 
python manage.py migrate 
python manage.py runserver 
``` 
### **5️ Access the Web App**   
Open **http://1️2️7.0.0.1️:8000/** in your browser. --- 
## **      
### **       
Usage**   
Text-Based Mood Detection**   
1
 ️ Navigate to `Detect Mood`   
2
 ️ Enter a short text or answer psychological questions   
3 AI (BERT Model) detects mood based on semantic similarity   
4
 ️ The detected mood is stored and analyzed   
### **   
Real-Time Facial Emotion Recognition**   
1
 ️ Navigate to `Detect Face`   
2
 ️ The camera captures **facial expressions, eye activity, and brightness**   
3 Haar cascades (`face`, `smile`, `eye`) classify moods in **real-time**   
4
 ️ The result is displayed and logged   
### **      
Mood Analysis Dashboard**   
1
 ️ Navigate to `Analyse Data`   
2
 ️ **Line Chart**     
3 **Bar Chart**       
### **      - Tracks mood changes over time   - Shows mood distribution   
AI-Based Task Suggestions**   
1
 ️ Based on detected mood, relevant **tasks** are suggested   --- 
## **     
Key Technologies Used**   
**Backend:** Django (Python)   
**Frontend:** HTML, CSS, JavaScript   
**Computer Vision:** OpenCV (Face, Smile, Eye Detection)   
**Machine Learning:** BERT + Cosine Similarity (Text Analysis)   
**Data Visualization:** Plotly.js   
**Database:** SQLite   
--- --- 
## **        --- 
## **     
Future Enhancements**   
**Sentiment Analysis with LLMs**   
**More Advanced Facial Expression Recognition**   
**Custom AI Models for Task Optimization**   
Contributors**   
**Param Parekh** - Developer   
Email: parammparekh1️3@gmail.com 
