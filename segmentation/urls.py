from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('segment/', views.segment_image, name='segment_image'),
    path('result/', views.result, name='result'),
    path('custom-fcm/', views.custom_fcm_segmentation, name='custom_fcm'),
] 