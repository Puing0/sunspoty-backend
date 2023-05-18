from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
	# path('directory/index.html', views.upload_file),
	path("", views.upload_file, name="homepage"),
    path("dbscanAlgo", views.dbscanAlgo, name="dbscanAlgo"),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)