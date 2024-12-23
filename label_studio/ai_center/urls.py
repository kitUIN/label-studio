"""This file and its contents are licensed under the Apache License 2.0. Please see the included NOTICE for copyright information and LICENSE for a copy of the license.
"""
from django.urls import path
from .views import login_node

urlpatterns = [
    path('api/ai/center/node/', login_node, name='ai_login_node'),
]
