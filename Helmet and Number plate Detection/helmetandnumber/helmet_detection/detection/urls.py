from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.CustomLoginView.as_view(template_name='login.html'), name='login'),
    path('register/', views.register, name='register'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('upload/', views.upload_video, name='upload'),
    path('stream/<str:filename>/', views.stream_video, name='stream'),
    path('stop/', views.stop_detection_view, name='stop'),
]