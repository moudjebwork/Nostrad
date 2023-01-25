from django.urls import path
from .views import index
from .views import upcoming
from .views import review

urlpatterns =[
    path('', index, name="mainpage-index"),
    path('upcoming/', upcoming, name="mainpage-upcoming"),
    path('review/', review, name="mainpage-review" )
]