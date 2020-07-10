from django.shortcuts import render
#import run

def index(request):
    return render(request,'index.html')

def config(request):
    return render(request, 'config.html')

def results(request):
    return render(request, 'results.html')

def run(request):
    return render(request, 'run')