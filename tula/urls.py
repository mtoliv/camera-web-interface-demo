from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from . import views

app_name = 'tula'
urlpatterns = [
    path('', views.HomeView, name='home'),
    path('menu', views.MenuView, name='menu'),
    path('message', views.MessageView, name='message'),
    path('headertopframe', views.HeaderTopFrameView, name='headertopframe'),
    path('edit/<file_name>', views.editview, name='edit'),
    path('edit_rejected/<file_name>', views.editview_rejected, name='edit_rejected'),
    path('switch/<file_name>', views.switch, name='switch'),
    path('switch_rejected/<file_name>', views.switch_rejected, name='switch_rejected'),
    path('train/', views.TrainView, name='train'),
    path('train/menutrain', views.MenuTrainView, name='menutrain'),
    path('real_time_view/', views.RealTimeView, name='real_time_view'),
    path('real_time_view/topframe', views.TopFrameView, name='topframe'),
    path('real_time_view/bottomframe', views.BottomFrameView, name='bottomframe'),
    path('real_time_view/menurtv', views.MenuRTVView, name='menurtv'),
    path('train/accepted_list/', views.accepted_list, name='accepted_list'),
    path('train/rejected_list/', views.rejected_list, name='rejected_list'),
    path('list_choice/', views.list_choice, name='list_choice'),
    path('real_time_view/recent', views.recent, name='recent'),
    path('train/topframetrain', views.TopFrameTrainView, name='traintopframe'),
    path('train/start_demo', views.start_demo, name='start_demo'),
    path('train/stop_demo', views.stop_demo, name='stop_demo'),
    path('train/start_classifier', views.start_classifier, name='start_classifier'),
    path('train/empty_images', views.empty_images, name='empty_images'),
]