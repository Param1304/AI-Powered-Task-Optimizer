from django.urls import path
from .views import home, detect_mood, suggest_task, analyse_data, detect_face, video_feed
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', home, name='home'),
    path('detect-mood/', detect_mood, name='detect_mood'),
    path('suggest-task/<str:mood>/', suggest_task, name='suggest_task'),
    path('analyse-data/', analyse_data, name='analyse_data'),
    path('detect_face/', detect_face, name='detect_face'),
    path('video_feed/', video_feed, name='video_feed'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)