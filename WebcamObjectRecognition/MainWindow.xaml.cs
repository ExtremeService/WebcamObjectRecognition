using ICSharpCode.SharpZipLib.GZip;
using Ionic.BZip2;
using IronPython.Zlib;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;
using ICSharpCode.SharpZipLib.Core;
using ICSharpCode.SharpZipLib.Tar;


namespace WebcamObjectRecognition
{
    public partial class MainWindow : System.Windows.Window
    {
        private VideoCapture _capture;
        private bool _isRunning;
        private bool _trainMode;
        private bool _detectMode;
        private string _label;
        private MLContext _mlContext;
        private PredictionEngine<ImageData, ImagePrediction> _predictionEngine;
        private const int IMG_SIZE = 224; // ResNet expects 224x224
        private const string DATA_DIR = "training_data";

        public MainWindow()
        {
            InitializeComponent();
            InitializeCamera();
            _isRunning = true;
            _trainMode = false;
            _detectMode = false;
            _label = "";
            _mlContext = new MLContext();
            Task.Run(() => ProcessCameraFeed());
        }

        private void InitializeCamera()
        {
            _capture = new VideoCapture(0);
            if (!_capture.IsOpened())
            {
                MessageBox.Show("Error: Could not open camera.");
                Close();
            }
        }




         void CreateModel()
        {
            string assetsPath = @"C:\Temp\MLTraining\assets";

            string outputMlNetModelFilePath = Path.Combine(assetsPath, "outputs", "imageClassifier.zip");
            string imagesFolderPathForPredictions = Path.Combine(assetsPath, "inputs", "test-images");

            string imagesDownloadFolderPath = Path.Combine(assetsPath, "inputs", "images");

            // 1. Download the image set and unzip
            string finalImagesFolderName = DownloadImageSet(imagesDownloadFolderPath);
            string fullImagesetFolderPath = Path.Combine(imagesDownloadFolderPath, finalImagesFolderName);

            _mlContext = new MLContext(seed: 1);

            // Specify MLContext Filter to only show feedback log/traces about ImageClassification
            // This is not needed for feedback output if using the explicit MetricsCallback parameter
            _mlContext.Log += FilterMLContextLog;

            // 2. Load the initial full image-set into an IDataView and shuffle so it'll be better balanced
            IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: fullImagesetFolderPath, useFolderNameAsLabel: true);
            IDataView fullImagesDataset = _mlContext.Data.LoadFromEnumerable(images);
            IDataView shuffledFullImageFilePathsDataset = _mlContext.Data.ShuffleRows(fullImagesDataset);

            // 3. Load Images with in-memory type within the IDataView and Transform Labels to Keys (Categorical)
            IDataView shuffledFullImagesDataset = _mlContext.Transforms.Conversion.
                    MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label", keyOrdinality: KeyOrdinality.ByValue)
                .Append(_mlContext.Transforms.LoadRawImageBytes(
                                                outputColumnName: "Image",
                                                imageFolder: fullImagesetFolderPath,
                                                inputColumnName: "ImagePath"))
                .Fit(shuffledFullImageFilePathsDataset)
                .Transform(shuffledFullImageFilePathsDataset);

            // 4. Split the data 80:20 into train and test sets, train and evaluate.
            var trainTestData = _mlContext.Data.TrainTestSplit(shuffledFullImagesDataset, testFraction: 0.2);
            IDataView trainDataView = trainTestData.TrainSet;
            IDataView testDataView = trainTestData.TestSet;

            // 5. Define the model's training pipeline using DNN default values
            //
            var pipeline = _mlContext.MulticlassClassification.Trainers
                    .ImageClassification(featureColumnName: "Image",
                                         labelColumnName: "LabelAsKey",
                                         validationSet: testDataView)
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel",
                                                                      inputColumnName: "PredictedLabel"));

            // 6. Train/create the ML model
            Console.WriteLine("*** Training the image classification model with DNN Transfer Learning on top of the selected pre-trained model/architecture ***");

            // Measuring training time
            var watch = Stopwatch.StartNew();

            //Train
            ITransformer trainedModel = pipeline.Fit(trainDataView);

            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;

            Console.WriteLine($"Training with transfer learning took: {elapsedMs / 1000} seconds");

            // 7. Get the quality metrics (accuracy, etc.)
            EvaluateModel(_mlContext, testDataView, trainedModel);

            // 8. Save the model to assets/outputs (You get ML.NET .zip model file and TensorFlow .pb model file)
            _mlContext.Model.Save(trainedModel, trainDataView.Schema, outputMlNetModelFilePath);

            // 9. Try a single prediction simulating an end-user app
            TrySinglePrediction(imagesFolderPathForPredictions, _mlContext, trainedModel);

        }

        private static void EvaluateModel(MLContext mlContext, IDataView testDataset, ITransformer trainedModel)
        {
            Console.WriteLine("Making predictions in bulk for evaluating model's quality...");

            // Measuring time
            var watch = Stopwatch.StartNew();

            var predictionsDataView = trainedModel.Transform(testDataset);

            var metrics = mlContext.MulticlassClassification.Evaluate(predictionsDataView, labelColumnName: "LabelAsKey", predictedLabelColumnName: "PredictedLabel");
            //ConsoleHelper.PrintMultiClassClassificationMetrics("TensorFlow DNN Transfer Learning", metrics);

            watch.Stop();
            var elapsed2Ms = watch.ElapsedMilliseconds;

            Console.WriteLine($"Predicting and Evaluation took: {elapsed2Ms / 1000} seconds");
        }

        private static void TrySinglePrediction(string imagesFolderPathForPredictions, MLContext mlContext, ITransformer trainedModel)
        {
            // Create prediction function to try one prediction
            var predictionEngine = mlContext.Model.CreatePredictionEngine<InMemoryImageData, ImagePrediction>(trainedModel);

            var testImages = FileUtils.LoadInMemoryImagesFromDirectory(imagesFolderPathForPredictions, false);

            var imageToPredict = testImages.First();

            var prediction = predictionEngine.Predict(imageToPredict);
        }


        public static IEnumerable<ImageData> LoadImagesFromDirectory( string folder, bool useFolderNameAsLabel = true) => FileUtils.LoadImagesFromDirectory(folder, useFolderNameAsLabel).Select(x => new ImageData(x.imagePath, x.label));


        public static string DownloadImageSet(string imagesDownloadFolder)
        {
            // get a set of images to teach the network about the new classes

            //SINGLE SMALL FLOWERS IMAGESET (200 files)
            const string fileName = "flower_photos_small_set.zip";
            var url = $"https://aka.ms/mlnet-resources/datasets/flower_photos_small_set.zip";
            Web.Download(url, imagesDownloadFolder, fileName);
            Compress.UnZip(Path.Join(imagesDownloadFolder, fileName), imagesDownloadFolder);

            //SINGLE FULL FLOWERS IMAGESET (3,600 files)
            //string fileName = "flower_photos.tgz";
            //string url = $"http://download.tensorflow.org/example_images/{fileName}";
            //Web.Download(url, imagesDownloadFolder, fileName);
            //Compress.ExtractTGZ(Path.Join(imagesDownloadFolder, fileName), imagesDownloadFolder);

            return Path.GetFileNameWithoutExtension(fileName);
        }

        public static void ConsoleWriteImagePrediction(string ImagePath, string Label, string PredictedLabel, float Probability)
        {
            var defaultForeground = Console.ForegroundColor;
            var labelColor = ConsoleColor.Magenta;
            var probColor = ConsoleColor.Blue;

            Console.Write("Image File: ");
            Console.ForegroundColor = labelColor;
            Console.Write($"{Path.GetFileName(ImagePath)}");
            Console.ForegroundColor = defaultForeground;
            Console.Write(" original labeled as ");
            Console.ForegroundColor = labelColor;
            Console.Write(Label);
            Console.ForegroundColor = defaultForeground;
            Console.Write(" predicted as ");
            Console.ForegroundColor = labelColor;
            Console.Write(PredictedLabel);
            Console.ForegroundColor = defaultForeground;
            Console.Write(" with score ");
            Console.ForegroundColor = probColor;
            Console.Write(Probability);
            Console.ForegroundColor = defaultForeground;
            Console.WriteLine("");
        }

        private static void FilterMLContextLog(object sender, LoggingEventArgs e)
        {
            if (e.Message.StartsWith("[Source=ImageClassificationTrainer;"))
            {
                Console.WriteLine(e.Message);
            }
        }
















        private async void ProcessCameraFeed()
        {
            while (_isRunning)
            {
                using var frame = _capture.RetrieveMat();
                if (frame.Empty())
                    continue;

                var bitmap = frame.ToBitmapSource();
                bitmap.Freeze();
                Dispatcher.Invoke(() =>
                {
                    WebcamImage.Source = bitmap;

                    if (_detectMode && _predictionEngine != null)
                    {
                        var prediction = PredictImage(frame);
                        float maxScore = prediction.Score?.Max() ?? 0f;
                        if (maxScore > 0.7f)
                        {
                            StatusText.Text = $"Detected: {prediction.Label} ({maxScore:F2})";
                            StatusText.Foreground = System.Windows.Media.Brushes.Red;
                        }
                        else
                        {
                            StatusText.Text = "Detect Mode";
                            StatusText.Foreground = System.Windows.Media.Brushes.Green;
                        }
                    }
                });
                await Task.Delay(33); // ~30 FPS
            }
        }

        private void TrainButton_Click(object sender, RoutedEventArgs e)
        {
            if (!_trainMode)
            {
                if (string.IsNullOrWhiteSpace(LabelInput.Text))
                {
                    MessageBox.Show("Please enter a label (e.g., superman).");
                    return;
                }
                _label = LabelInput.Text.Trim();
                _trainMode = true;
                _detectMode = false;
                StatusText.Text = $"Train Mode: {_label}";
                StatusText.Foreground = System.Windows.Media.Brushes.Green;
                LabelInput.IsEnabled = false;
                CaptureButton.IsEnabled = true;
                TrainButton.Content = "Stop Train Mode";
                DetectButton.IsEnabled = false;
            }
            else
            {
                _trainMode = false;
                StatusText.Text = "Idle";
                StatusText.Foreground = System.Windows.Media.Brushes.Green;
                LabelInput.IsEnabled = true;
                CaptureButton.IsEnabled = false;
                TrainButton.Content = "Start Train Mode";
                DetectButton.IsEnabled = true;
            }
        }

        private void CaptureButton_Click(object sender, RoutedEventArgs e)
        {
            if (!_trainMode)
                return;

            using var frame = _capture.RetrieveMat();
            if (frame.Empty())
                return;

            var labelDir = Path.Combine(DATA_DIR, _label);
            Directory.CreateDirectory(labelDir);

            var timestamp = DateTime.Now.Ticks;
            var imgPath = Path.Combine(labelDir, $"{timestamp}.jpg");
            frame.ImWrite(imgPath);
            Dispatcher.Invoke(() => StatusText.Text = $"Captured: {_label}");
        }

        private void DetectButton_Click(object sender, RoutedEventArgs e)
        {
            if (!_detectMode)
            {
                _detectMode = true;
                _trainMode = false;
                StatusText.Text = "Training...";
                StatusText.Foreground = System.Windows.Media.Brushes.Green;
                LabelInput.IsEnabled = false;
                CaptureButton.IsEnabled = false;
                TrainButton.IsEnabled = false;
                DetectButton.Content = "Stop Detect Mode";
                CreateModel();
                Task.Run(() => TrainModel());
            }
            else
            {
                _detectMode = false;
                StatusText.Text = "Idle";
                StatusText.Foreground = System.Windows.Media.Brushes.Green;
                LabelInput.IsEnabled = true;
                CaptureButton.IsEnabled = false;
                TrainButton.IsEnabled = true;
                DetectButton.Content = "Start Detect Mode";
                _predictionEngine?.Dispose();
                _predictionEngine = null;
            }
        }

        private void StopButton_Click(object sender, RoutedEventArgs e)
        {
            _isRunning = false;
            _capture?.Release();
            _predictionEngine?.Dispose();
            Close();
        }

        private void TrainModel()
        {
            try
            {

            }
            catch (Exception ex)
            {
                Dispatcher.Invoke(() =>
                {
                    MessageBox.Show($"Training failed: {ex.Message}");
                    ResetToIdle();
                });
            }
        }


        private ImagePrediction PredictImage(Mat img)
        {
            try
            {
                return new ImagePrediction { Label = "Unknown", Score = new float[] { 0f } };
            }
            catch
            {
                return new ImagePrediction { Label = "Unknown", Score = new float[] { 0f } };
            }
        }

        private void ResetToIdle()
        {
            _detectMode = false;
            _trainMode = false;
            StatusText.Text = "Idle";
            StatusText.Foreground = System.Windows.Media.Brushes.Green;
            LabelInput.IsEnabled = true;
            CaptureButton.IsEnabled = false;
            TrainButton.IsEnabled = true;
            DetectButton.Content = "Start Detect Mode";
            DetectButton.IsEnabled = true;
            _predictionEngine?.Dispose();
            _predictionEngine = null;
        }
    }

    public class FileUtils
    {
        public static IEnumerable<(string imagePath, string label)> LoadImagesFromDirectory(
            string folder,
            bool useFolderNameasLabel)
        {
            var imagesPath = Directory
                .GetFiles(folder, "*", searchOption: SearchOption.AllDirectories)
                .Where(x => Path.GetExtension(x) == ".jpg" || Path.GetExtension(x) == ".png");

            return useFolderNameasLabel
                ? imagesPath.Select(imagePath => (imagePath, Directory.GetParent(imagePath).Name))
                : imagesPath.Select(imagePath =>
                {
                    var label = Path.GetFileName(imagePath);
                    for (var index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                    return (imagePath, label);
                });
        }

        public static IEnumerable<InMemoryImageData> LoadInMemoryImagesFromDirectory(
            string folder,
            bool useFolderNameAsLabel = true)
            => LoadImagesFromDirectory(folder, useFolderNameAsLabel)
                .Select(x => new InMemoryImageData(
                    image: File.ReadAllBytes(x.imagePath),
                    label: x.label,
                    imageFileName: Path.GetFileName(x.imagePath)));

        public static string GetAbsolutePath(Assembly assembly, string relativePath)
        {
            var assemblyFolderPath = new FileInfo(assembly.Location).Directory.FullName;

            return Path.Combine(assemblyFolderPath, relativePath);
        }
    }



    public class ImageData
    {

        public ImageData(string imagePath, string label)
        {
            ImagePath = imagePath;
            Label = label;
        }

        public string ImagePath { get; set; }
        public byte[] ImageBytes { get; set; }
        public string Label { get; set; }
    }

    public class ImagePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Label { get; set; }
        public float[] Score { get; set; }
    }

    public static class OpenCvSharpExtensions
    {
        public static BitmapSource ToBitmapSource(this Mat mat)
        {
            using var ms = mat.ToMemoryStream();
            var bitmap = new System.Windows.Media.Imaging.BitmapImage();
            bitmap.BeginInit();
            bitmap.StreamSource = ms;
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.EndInit();
            bitmap.Freeze();
            return bitmap;
        }
    }

    public class InMemoryImageData
    {
        public InMemoryImageData(byte[] image, string label, string imageFileName)
        {
            Image = image;
            Label = label;
            ImageFileName = imageFileName;
        }

        public readonly byte[] Image;

        public readonly string Label;

        public readonly string ImageFileName;
    }




    public class Compress
    {
        public static void ExtractGZip(string gzipFileName, string targetDir)
        {
            // Use a 4K buffer. Any larger is a waste.    
            byte[] dataBuffer = new byte[4096];

            using (System.IO.Stream fs = new FileStream(gzipFileName, FileMode.Open, FileAccess.Read))
            {
                using (GZipInputStream gzipStream = new GZipInputStream(fs))
                {
                    // Change this to your needs
                    string fnOut = Path.Combine(targetDir, Path.GetFileNameWithoutExtension(gzipFileName));

                    using (FileStream fsOut = File.Create(fnOut))
                    {
                        StreamUtils.Copy(gzipStream, fsOut, dataBuffer);
                    }
                }
            }
        }

        public static void UnZip(String gzArchiveName, String destFolder)
        {
            var flag = gzArchiveName.Split(Path.DirectorySeparatorChar).Last().Split('.').First() + ".bin";
            if (File.Exists(Path.Combine(destFolder, flag))) return;

            Console.WriteLine($"Extracting.");
            var task = Task.Run(() =>
            {
                ZipFile.ExtractToDirectory(gzArchiveName, destFolder);
            });

            while (!task.IsCompleted)
            {
                Thread.Sleep(200);
                Console.Write(".");
            }

            File.Create(Path.Combine(destFolder, flag));
            Console.WriteLine("");
            Console.WriteLine("Extracting is completed.");
        }

        public static void ExtractTGZ(String gzArchiveName, String destFolder)
        {
            var flag = gzArchiveName.Split(Path.DirectorySeparatorChar).Last().Split('.').First() + ".bin";
            if (File.Exists(Path.Combine(destFolder, flag))) return;

            Console.WriteLine($"Extracting.");
            var task = Task.Run(() =>
            {
                using (var inStream = File.OpenRead(gzArchiveName))
                {
                    using (var gzipStream = new GZipInputStream(inStream))
                    {
                        using (TarArchive tarArchive = TarArchive.CreateInputTarArchive(gzipStream))
                            tarArchive.ExtractContents(destFolder);
                    }
                }
            });

            while (!task.IsCompleted)
            {
                Thread.Sleep(200);
                Console.Write(".");
            }

            File.Create(Path.Combine(destFolder, flag));
            Console.WriteLine("");
            Console.WriteLine("Extracting is completed.");
        }
    }
    public class Web
    {
        public static bool Download(string url, string destDir, string destFileName)
        {
            if (destFileName == null)
                destFileName = url.Split(Path.DirectorySeparatorChar).Last();

            Directory.CreateDirectory(destDir);

            string relativeFilePath = Path.Combine(destDir, destFileName);

            if (File.Exists(relativeFilePath))
            {
                Console.WriteLine($"{relativeFilePath} already exists.");
                return false;
            }

            var wc = new WebClient();
            Console.WriteLine($"Downloading {relativeFilePath}");
            var download = Task.Run(() => wc.DownloadFile(url, relativeFilePath));
            while (!download.IsCompleted)
            {
                Thread.Sleep(1000);
                Console.Write(".");
            }
            Console.WriteLine("");
            Console.WriteLine($"Downloaded {relativeFilePath}");

            return true;
        }
    }

}