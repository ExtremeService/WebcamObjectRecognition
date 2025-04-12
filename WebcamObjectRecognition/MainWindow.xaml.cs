using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;

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
                var imageData = LoadTrainingData();
                if (!imageData.Any())
                {
                    Dispatcher.Invoke(() =>
                    {
                        MessageBox.Show("No training images found. Please capture images first.");
                        ResetToIdle();
                    });
                    return;
                }

                var dataView = _mlContext.Data.LoadFromEnumerable(imageData);

                // ML.NET 3.0.0 pipeline
                var pipeline = _mlContext.Transforms.LoadImages("Image", null, "ImagePath")
                    .Append(_mlContext.Transforms.ResizeImages("Image", IMG_SIZE, IMG_SIZE))
                    .Append(_mlContext.Transforms.ExtractPixels("Image"))
                    .Append(_mlContext.Transforms.Conversion.MapValueToKey("Label", "Label"))
                    .Append(_mlContext.MulticlassClassification.Trainers.ImageClassification(
                        new ImageClassificationTrainer.Options
                        {
                            FeatureColumnName = "Image",
                            LabelColumnName = "Label",
                            Arch = ImageClassificationTrainer.Architecture.ResnetV250,
                            Epoch = 20,
                            BatchSize = 10,
                            ValidationSet = _mlContext.Data.TrainTestSplit(dataView, 0.2).TestSet
                        }))
                    .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                var model = pipeline.Fit(dataView);
                _predictionEngine = _mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);

                Dispatcher.Invoke(() => StatusText.Text = "Detect Mode");
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

        private IEnumerable<ImageData> LoadTrainingData()
        {
            var images = new List<ImageData>();
            if (!Directory.Exists(DATA_DIR))
                return images;

            foreach (var labelDir in Directory.GetDirectories(DATA_DIR))
            {
                var label = Path.GetFileName(labelDir);
                foreach (var imgPath in Directory.GetFiles(labelDir, "*.jpg"))
                {
                    if (File.Exists(imgPath))
                        images.Add(new ImageData { ImagePath = imgPath, Label = label });
                }
            }
            return images;
        }

        private ImagePrediction PredictImage(Mat img)
        {
            try
            {
                using var resized = img.Resize(new OpenCvSharp.Size(IMG_SIZE, IMG_SIZE));
                using var ms = resized.ToMemoryStream();
                var imageData = new ImageData { ImagePath = "temp", ImageBytes = ms.ToArray() };
                return _predictionEngine.Predict(imageData);
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

    public class ImageData
    {
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
}