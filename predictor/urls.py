from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('clear-history/', views.clear_history, name='clear_history'),
    path('api/explain/', views.api_explain, name='api_explain'),
    path('api/chat/', views.api_chat, name='api_chat'),
]
