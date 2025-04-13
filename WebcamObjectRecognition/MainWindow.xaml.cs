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
using static ImageClassification.MLModel;
using ImageClassification;
using ImageClassification.DataModels;
using System.Windows.Media;


namespace WebcamObjectRecognition
{
    public partial class MainWindow : System.Windows.Window
    {
        private bool _isRunning;
        private bool _trainMode;
        private bool _detectMode;
        private string _label;
        private MLContext _mlContext;
        private PredictionEngine<ImageData, ImagePrediction> _predictionEngine;

        public MainWindow()
        {
            InitializeComponent();
            _isRunning = true;
            _trainMode = false;
            _detectMode = false;
            _label = "";
            _mlContext = new MLContext();
            Task.Run(() => ProcessCameraFeed());
            CreateDirectories();
        }

        private async void ProcessCameraFeed()
        {
            while (_isRunning)
            {
                if (_capture == null) 
                {
                    InitializeCamera();
                }
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
                    var a = LabelInput.Background;
                    LabelInput.BorderBrush = new SolidColorBrush(Colors.Red);
                    return;
                }
                LabelInput.Background = System.Windows.Media.Brushes.Green;
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
            TakePictures(_label,10,200);
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
                var a  = TakeSinglePicturesForPrediction();
                var prediction = LoadModelandPredict(a[0]);


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

        private void Train_Click(object sender, RoutedEventArgs e)
        {
            MLModel.outputMlNetModelFileName = "MLModel.zip";
            Task.Run(() => CreateModel());
        }

        private void TextBox_TextChanged(object sender, System.Windows.Controls.TextChangedEventArgs e)
        {

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
    }
}