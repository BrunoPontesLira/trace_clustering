from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include('trace_clustering.urls')),
    path('admin/', admin.site.urls),
]
