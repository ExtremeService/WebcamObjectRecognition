This WFP Application allows it to capture labeled pictures with an UVC-compliant Camera (Mode: Capture), then train a model (Mode: Train).
Change to Detection mode  (Mode: Detection) and the SW predicts the label for wathever is captured by the camera.

HW Requirement: 

UVC Camera
- IPEVO V4K Ultra High Definition Document Camera (100$)
  good results, slightly rotate camera to generate training pictures.

- Logitech C925e BUSINESS WEBCAM (140$)
  Difficult to train due to missing stativ 




SW Requirement:

Windows, Microsoft Visual Studio Community 2022 (64-bit) 

Uses ML.NET with OpenCV. Model selection InceptionV3, feel free to change to InceptionV3/MobilenetV2/ResnetV250

All data is stored under the hardcoded path: C:\Temp\MLTraining\assets

Sources:
https://x.com/i/grok
https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/iris-clustering
https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/image-classification-api-transfer-learning
https://github.com/dotnet/machinelearning-samples/
https://www.rheinwerk-verlag.de/neuronale-netze-programmieren-mit-python/
  
  
