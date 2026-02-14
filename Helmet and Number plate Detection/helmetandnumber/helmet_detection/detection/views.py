from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import RegisterForm, UploadForm
from django.conf import settings
from django.http import StreamingHttpResponse
from .helmetnum import generate_frames, stop_stream
import os

# 🟥 Stop detection when button clicked
def stop_detection_view(request):
    stop_stream()
    return redirect("upload")


# 🟨 Register new user
def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = RegisterForm()
    return render(request, 'register.html', {'form': form})


# 🟩 Upload and process video
@login_required
def upload_video(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_file = form.cleaned_data['video']
            upload_path = os.path.join(settings.MEDIA_ROOT, 'uploads', video_file.name)

            os.makedirs(os.path.dirname(upload_path), exist_ok=True)
            with open(upload_path, 'wb+') as f:
                for chunk in video_file.chunks():
                    f.write(chunk)

            # ✅ Render the result page (not redirect to stream)
            return render(request, "result.html", {"filename": video_file.name})
    else:
        form = UploadForm()
    return render(request, 'upload.html', {'form': form})


# 🟦 Stream YOLO processed frames
@login_required
def stream_video(request, filename):
    video_path = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)

    return StreamingHttpResponse(
        generate_frames(video_path),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

from django.contrib.auth import logout
from django.contrib.auth.views import LoginView

class CustomLoginView(LoginView):
    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            logout(request)  # ✅ logs out if they were still logged in
        return super().dispatch(request, *args, **kwargs)

