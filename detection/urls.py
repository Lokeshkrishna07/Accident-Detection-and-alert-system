from django.urls import path
from .views import index , VideoUploadView

urlpatterns = [
    path('', index, name='home'),
    path('api/upload/', VideoUploadView.as_view(), name='video-upload'),
]