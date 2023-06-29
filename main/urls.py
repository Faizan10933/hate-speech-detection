from django.contrib import admin
from . import views
from django.urls import path

urlpatterns = [
    path('', views.detect, name="detect"),
    # path('search/', views.search, name="search"),
    # path('results/', views.results, name="results")
]