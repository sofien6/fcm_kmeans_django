from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('segment/', views.segment_image, name='segment_image'),
    path('result/<str:filename>/', views.result, name='result'),
] 