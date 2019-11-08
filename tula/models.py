import datetime

from django.db import models
from django.utils import timezone

class File(models.Model):
	file_name = models.CharField(max_length=200)
	status = models.CharField(max_length=200)
	def __str__(self):
		return self.file_name
	def create(file_n,stat):
		f = File(file_name=file_n,status=stat)
		f.save()