# views.py
import requests
from django.contrib.auth import authenticate, login
from django.http import JsonResponse
from django.shortcuts import redirect, render

from users.views import user_login
import base64


def login_node(request, *args, **kwargs):
    return render(request, "to_login.html", {
        "url": base64.b64decode(request.GET.get("url", "L3VzZXIvbG9naW4v")).decode()
    })
