using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Common;
using ImageClassification.DataModels;
using Microsoft.ML;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;
using OpenCvSharp;
using System.Threading;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using Google.Protobuf;
using Microsoft.ML.Vision;
using System.Windows;

namespace ImageClassification
{
    public class MLModel
    {

        //https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/iris-clustering
        //https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/image-classification-api-transfer-learning
        //https://github.com/dotnet/machinelearning-samples/

        static string assetsPath = @"C:\Temp\MLTraining\assets";

        static string InputImageReference { get => Path.Combine(assetsPath, "inputs", "reference.jpg"); set => throw new NotImplementedException(); }
        static string outputMlNetModelFilePath { get => Path.Combine(assetsPath, "outputs", outputMlNetModelFileName); set => throw new NotImplementedException(); }

        public static string outputMlNetModelFileName { get; set; } = "MLModel.zip";
        static string imagesFolderPathForPredictions { get; set; }  // store the test images, which are then predicted   
        public static string imagesFolderPathForTraining { get; set; } // store the training material

        public static List<string> NewPicturesTaken  { get; set; } = new List<string>();

        public static bool DetectionRunning { get; set; } 

        static PredictionEngine<InMemoryImageData, ImagePrediction> predictionEngine;

        static MLContext mlContext = new MLContext(seed: 1); // seed for reproducibility

        static ITransformer trainedModel = null;
        public  static VideoCapture _capture;
        public static void CreateDirectories()
        {
            Directory.CreateDirectory(assetsPath);
            Directory.CreateDirectory(Path.GetDirectoryName(outputMlNetModelFilePath));
            imagesFolderPathForPredictions = Path.Combine(assetsPath, "inputs", "test-images"); // store the test images, which are then predicted   
            imagesFolderPathForTraining =       Path.Combine(assetsPath, "inputs", "images"); // store the training material
            Directory.CreateDirectory(imagesFolderPathForPredictions);
            Directory.CreateDirectory(imagesFolderPathForTraining);

        }


        public static bool NormalizeToReferenceImage(string imagePath, ConcurrentQueue<string> mlTraining_status)
        {
            bool result = false;
            if (!File.Exists(InputImageReference) )
            {
                mlTraining_status.Enqueue($"Reference Picture for orientation not found, please create under path: {InputImageReference} ");
                return result;
            }
            var referenceImage = Cv2.ImRead(InputImageReference, ImreadModes.Grayscale);
            var inputImage = Cv2.ImRead(imagePath, ImreadModes.Grayscale);
            var coloredInputImage = Cv2.ImRead(imagePath, ImreadModes.Color); // For final warping

            // Step 1: Detect keypoints and descriptors using ORB
            var orb = ORB.Create(nFeatures: 1000);
            KeyPoint[] refKeypoints, inputKeypoints;
            var refDescriptors = new Mat();
            var inputDescriptors = new Mat();
            orb.DetectAndCompute(referenceImage, null, out refKeypoints, refDescriptors);
            orb.DetectAndCompute(inputImage, null, out inputKeypoints, inputDescriptors);

            // Step 2: Match descriptors using BFMatcher
            var matcher = new BFMatcher(NormTypes.Hamming, crossCheck: true);
            var matches = matcher.Match(refDescriptors, inputDescriptors);

            // Step 3: Filter good matches (optional: apply ratio test)
            matches = matches.OrderBy(x => x.Distance).Take(100).ToArray(); // Take top 100 matches

            // Step 4: Extract matched keypoints
            var refPoints = new Point2f[matches.Length];
            var inputPoints = new Point2f[matches.Length];
            for (int i = 0; i < matches.Length; i++)
            {
                refPoints[i] = refKeypoints[matches[i].QueryIdx].Pt;
                inputPoints[i] = inputKeypoints[matches[i].TrainIdx].Pt;
            }

            // Convert Point2f[] to Mat
            var inputPointsMat = Mat.FromArray(inputPoints); // Create Mat from inputPoints
            var refPointsMat = Mat.FromArray(refPoints);     // Create Mat from refPoints

            // Step 5: Estimate affine transformation
            var transform = Cv2.EstimateAffine2D(inputPointsMat, refPointsMat, null, RobustEstimationAlgorithms.RANSAC);

            if (transform == null)
            {
                Console.WriteLine("Could not compute affine transformation.");
                return result;
            }

            // Step 6: Warp the input image to align with the reference
            var warpedImage = new Mat();
            Cv2.WarpAffine(coloredInputImage, warpedImage, transform, referenceImage.Size());

            // Save or display the result
            Cv2.ImWrite(imagePath, warpedImage);
            result = true;
            return result;
        }


        public static void CreateModel(ConcurrentQueue<string> mlTraining_status)
        {

            // Specify MLContext Filter to only show feedback log/traces about ImageClassification
            // This is not needed for feedback output if using the explicit MetricsCallback parameter
            mlContext.Log += FilterMLContextLog;

            mlTraining_status.Enqueue("0% starting ML.NET image classification training...");
            // 2. Load the initial full image-set into an IDataView and shuffle so it'll be better balanced
            IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: imagesFolderPathForTraining, useFolderNameAsLabel: true);
            IDataView fullImagesDataset = mlContext.Data.LoadFromEnumerable(images);
            IDataView shuffledFullImageFilePathsDataset = mlContext.Data.ShuffleRows(fullImagesDataset);

            // 3. Load Images with in-memory type within the IDataView and Transform Labels to Keys (Categorical)
            IDataView shuffledFullImagesDataset = mlContext.Transforms.Conversion.
                    MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label", keyOrdinality: KeyOrdinality.ByValue)
                .Append(mlContext.Transforms.LoadRawImageBytes(
                                                outputColumnName: "Image",
                                                imageFolder: imagesFolderPathForTraining,
                                                inputColumnName: "ImagePath"))
                .Fit(shuffledFullImageFilePathsDataset)
                .Transform(shuffledFullImageFilePathsDataset);
            mlTraining_status.Enqueue($"20% Load Images with in-memory type within the IDataView and Transform Labels to Keys (Categorical)");


            // 4. Split the data 80:20 into train and test sets, train and evaluate.
            var trainTestData = mlContext.Data.TrainTestSplit(shuffledFullImagesDataset, testFraction: 0.2);
            IDataView trainDataView = trainTestData.TrainSet;
            IDataView testDataView = trainTestData.TestSet;

            // 5a Define the model's training pipeline using DNN default values

            //var pipeline = mlContext.MulticlassClassification.Trainers
            //        .ImageClassification(featureColumnName: "Image",
            //                             labelColumnName: "LabelAsKey",
            //                             validationSet: testDataView)
            //    .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel",
            //                                                          inputColumnName: "PredictedLabel"));

            //5b Define the model's training pipeline by using explicit hyper-parameters


            var options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                // Just by changing/selecting InceptionV3/MobilenetV2/ResnetV250  
                // you can try a different DNN architecture (TensorFlow pre-trained model). 
                Arch = ImageClassificationTrainer.Architecture.InceptionV3,
                Epoch = 50,       //100
                BatchSize = 10,
                LearningRate = 0.01f,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                ValidationSet = testDataView
            };

            var pipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(options)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                        outputColumnName: "PredictedLabel",
                        inputColumnName: "PredictedLabel"));

            // 6. Train/create the ML model
            Console.WriteLine("*** Training the image classification model with DNN Transfer Learning on top of the selected pre-trained model/architecture ***");
            mlTraining_status.Enqueue("60% DNN Transfer Learning");
            // Measuring training time
            var watch = Stopwatch.StartNew();

            //Train
            trainedModel = pipeline.Fit(trainDataView);

            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;

            Console.WriteLine($"Training with transfer learning took: {elapsedMs / 1000} seconds");
            mlTraining_status.Enqueue($"80% Training with transfer learning took: {elapsedMs / 1000} seconds");
            // 7. Get the quality metrics (accuracy, etc.)
            EvaluateModel(mlContext, testDataView, trainedModel);

            // 8. Save the model to assets/outputs (You get ML.NET .zip model file and TensorFlow .pb model file)
            mlContext.Model.Save(trainedModel, trainDataView.Schema, outputMlNetModelFilePath);
            mlTraining_status.Enqueue($"90% saving the model under {outputMlNetModelFilePath} ");

            // 9. Try a single prediction simulating an end-user app
            TryPredictionForFolder(imagesFolderPathForPredictions, mlContext, trainedModel);
            mlTraining_status.Enqueue("100% Model computed");
            predictionEngine = null;
        }
       
        private static void EvaluateModel(MLContext mlContext, IDataView testDataset, ITransformer trainedModel)
        {
            Console.WriteLine("Making predictions in bulk for evaluating model's quality...");

            // Measuring time
            var watch = Stopwatch.StartNew();

            var predictionsDataView = trainedModel.Transform(testDataset);

            var metrics = mlContext.MulticlassClassification.Evaluate(predictionsDataView, labelColumnName:"LabelAsKey", predictedLabelColumnName: "PredictedLabel");
            ConsoleHelper.PrintMultiClassClassificationMetrics("TensorFlow DNN Transfer Learning", metrics);

            watch.Stop();
            var elapsed2Ms = watch.ElapsedMilliseconds;

            Console.WriteLine($"Predicting and Evaluation took: {elapsed2Ms / 1000} seconds");
        }

        public static void TryPredictionForFolder(string imagesFolderPathForPredictions, MLContext mlContext, ITransformer trainedModel, double treshold=0.8)
        {
            // Create prediction function to try one prediction
            var predictionEngine = mlContext.Model.CreatePredictionEngine<InMemoryImageData, ImagePrediction>(trainedModel);

            var testImages = FileUtils.LoadInMemoryImagesFromDirectory(imagesFolderPathForPredictions, false);

            foreach(var image in testImages)
            {
                var prediction = predictionEngine.Predict(image);

                bool predicitonConfidenceFull = prediction.Score.Max() > treshold;
                string response = predicitonConfidenceFull ? "High confidence" : "Low confidence prediction, please check the image.";

                Console.WriteLine($"Image Filename : [{image.ImageFileName}], " +
                        $"Scores : [{string.Join(";", prediction.Score)}], " +
                        $"Predicted Label : {prediction.PredictedLabel}  {response}");

            }
        }



        public static void LoadModel(ConcurrentQueue<string> mlTraining_status)
        {
            if (predictionEngine != null)
            {
                return;
            }

            using (var stream = new FileStream(outputMlNetModelFilePath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                trainedModel = mlContext.Model.Load(stream, out var modelInputSchema);
            }
            predictionEngine = mlContext.Model.CreatePredictionEngine<InMemoryImageData, ImagePrediction>(trainedModel);
            mlTraining_status.Enqueue($"loading model from {outputMlNetModelFilePath} ");
        }

        public static ImagePrediction LoadModelandPredict(string FullFilePath, double threshold, ConcurrentQueue<string> mlTraining_status)
        {
            LoadModel(mlTraining_status);
            var Image = FileUtils.LoadInMemorySingleImageFromDirectory(FullFilePath);
            ImagePrediction prediction = predictionEngine.Predict(Image);

            bool predicitonConfidenceFull = prediction.Score.Max() > threshold;

            mlTraining_status.Enqueue($"Predicted Label : {prediction.PredictedLabel} with score: {prediction.Score.Max():F2} ");
            return prediction;
        }



        public  static void InitializeCamera()
        {
            int frameWidth = 1280;
            int frameHeight = 720;
            _capture = new VideoCapture(0);
            if (!_capture.Set(VideoCaptureProperties.FrameWidth, frameWidth))
            {
               
            }
            if (!_capture.Set(VideoCaptureProperties.FrameHeight, frameHeight))
            {
                
            }
            if (!_capture.IsOpened())
            {
                throw new Exception("Camera not found");
            }
        }


        public static void FaceDetection()
        {
            string cascadePath = "Resources/haarcascade_frontalface_default.xml";
            string imagePath = "input.jpg"; // Replace with your image path
            string outputDir = "OutputFaces";

            // Create output directory if it doesn't exist
            Directory.CreateDirectory(outputDir);

            // Load Haar Cascade for face detection
            var faceCascade = new CascadeClassifier(cascadePath);

            // Load the image
            var image = Cv2.ImRead(imagePath);
            if (image.Empty())
            {
                Console.WriteLine("Failed to load image.");
                return;
            }

            // Convert to grayscale for detection
            var gray = new Mat();
            Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);

            // Detect faces
            var faces = faceCascade.DetectMultiScale(
                image: gray,
                scaleFactor: 1.1,
                minNeighbors: 5,
                minSize: new Size(30, 30)
            );

            Console.WriteLine($"Detected {faces.Length} face(s).");

            // Extract and save each face
            for (int i = 0; i < faces.Length; i++)
            {
                var face = faces[i];
                // Crop the face region from the original image
                var faceImage = new Mat(image, face);

                // Save the cropped face
                string outputPath = Path.Combine(outputDir, $"face_{i + 1}.jpg");
                faceImage.SaveImage(outputPath);
                Console.WriteLine($"Saved face {i + 1} to {outputPath}");

                // Optionally, draw rectangle around face on original image
                Cv2.Rectangle(image, face, Scalar.Red, 2);
            }

            // Save the image with detected faces (optional)
            string outputImagePath = Path.Combine(outputDir, "image_with_faces.jpg");
            image.SaveImage(outputImagePath);
            Console.WriteLine($"Saved image with detected faces to {outputImagePath}");
        }


        public static bool IsModelReady(ConcurrentQueue<string> Messages)
        {
            if (!File.Exists(outputMlNetModelFilePath))
            {
                Messages.Enqueue($"Model not found at {outputMlNetModelFilePath}");
                return false;
            }
            else
            {
                return true;
            }
        }
        public static async Task TakeSinglePicturesForPrediction(ConcurrentQueue<string> Messages, object _cameralock)
        {
            DetectionRunning = true;
            while (true && DetectionRunning)
            {
                Messages.Enqueue("----------------------------");
                NewPicturesTaken.Clear();
                await TakePictures("-", number: 1, sleepMs: 0, Messages, _cameralock);
                LoadModelandPredict(NewPicturesTaken[0], 0.8, Messages);
                await Task.Delay(2000);
            }
        }

        public static async Task TakePictures(string labelname, int number, int sleepMs, ConcurrentQueue<string> Messages, object _cameralock)
        {
            if (_capture == null)
            {
                InitializeCamera();
            }
            var labelDir = "";
            if (labelname == "-") // label - is reserved for prediction 
            {
                labelDir = Path.GetTempPath();
            }
            else
            {
                labelDir = Path.Combine(imagesFolderPathForTraining, labelname);
                Directory.CreateDirectory(labelDir);
            }

            for (int i = 0; i< number; i++)
            {
                lock (_cameralock)
                {
                    _capture.Grab();
                    var frame = _capture.RetrieveMat();
                    var timestamp = DateTime.Now.Ticks;
                    var imgPath = Path.Combine(labelDir, $"{timestamp}.jpg");
                    frame.ImWrite(imgPath);
                    if (!NormalizeToReferenceImage(imgPath, Messages))
                    {
                        return;
                    }
                    NewPicturesTaken.Add(imgPath);
                }
                if (number != 1)
                {
                    await Task.Delay(sleepMs);
                    Messages.Enqueue($"Pcitures {i} of total {number} taken");
                }
            }
            if (labelname != "-")
            {
                Messages.Enqueue($"100% Photos done");
            }
            else
            {
                Messages.Enqueue($"Photo taken");
            }
        }


        public static IEnumerable<ImageData> LoadImagesFromDirectory(
            string folder,
            bool useFolderNameAsLabel = true)
            => FileUtils.LoadImagesFromDirectory(folder, useFolderNameAsLabel)
                .Select(x => new ImageData(x.imagePath, x.label));


        private static void FilterMLContextLog(object sender, LoggingEventArgs e)
        {
            if (e.Message.StartsWith("[Source=ImageClassificationTrainer;"))
            {
                Console.WriteLine(e.Message);
            }
        }
    }
}

