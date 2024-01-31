from django.conf.urls import url
from employee import views

urlpatterns = [
    url('lung', views.lung, name='lung'),
]
