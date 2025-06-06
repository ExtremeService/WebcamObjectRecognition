﻿using ICSharpCode.SharpZipLib.GZip;
using Microsoft.ML;
using OpenCvSharp;
using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;
using ICSharpCode.SharpZipLib.Core;
using static ImageClassification.MLModel;
using ImageClassification;
using ImageClassification.DataModels;
using System.Windows.Media;
using System.Windows.Controls;
using System.Collections.Concurrent;


namespace WebcamObjectRecognition
{
    public partial class MainWindow : System.Windows.Window
    {
        private bool _isRunning;
        public object _cameralock = new();
        private bool _trainMode;
        private bool _detectMode;
        private string _label;
        private MLContext _mlContext;
        private PredictionEngine<ImageData, ImagePrediction> _predictionEngine;
        static ConcurrentQueue<string> Messages = new ConcurrentQueue<string>();
        private bool? useFaceDetection = false;
        private bool? useReferenceImage = false;
        public MainWindow()
        {
            InitializeComponent();
            _isRunning = true;
            _trainMode = false;
            _detectMode = false;
            _label = "";
            _mlContext = new MLContext();

            Messages.Enqueue("Startup");
            Task.Run(() => ProcessCameraFeed());
            Task.Run(() => UpdateOutputbox());
            CreateDirectories();
        }



        private async void UpdateOutputbox()
        {
            while (_isRunning)
            {
                // updating the OutputBox with the latest message
                Dispatcher.Invoke(() =>
                {
                    if (Messages.TryDequeue(out string status))
                    {
                        OutputBox.Text = status + Environment.NewLine + OutputBox.Text;
                    }
                });
            }
            }
        private async void ProcessCameraFeed()
        {
            while (_isRunning)
            {

                if (_capture == null) 
                {
                    InitializeCamera();
                }
                lock (_cameralock)
                {
                    using var frame = _capture.RetrieveMat();
                    if (frame.Empty())
                        continue;
                    var bitmap = frame.ToBitmapSource();
                    bitmap.Freeze();
                    Dispatcher.Invoke(() =>
                    {
                        WebcamImage.Source = bitmap;
                    });
                }
                await Task.Delay(33); // ~30 FPS
            }
        }

        private void CaptureButton_Click(object sender, RoutedEventArgs e)
        {
            var tempLabel = LabelInput.Text.Trim();
            Task.Run(() => TakePictures(tempLabel, 100, 200, Messages, _cameralock));
        }

        private void DetectButton_Click(object sender, RoutedEventArgs e)
        {
            if (!_detectMode)
            {

                if (!IsModelReady(Messages)) return;
                _detectMode = true;
                DetectButton.Content = "Stop Detect Mode";
                Task.Run(() => TakeSinglePicturesForPrediction(Messages, _cameralock));
            }
            else
            {
                _detectMode = false;
                LabelInput.IsEnabled = true;
                CaptureButton.IsEnabled = false;
                DetectButton.Content = "Start Detect Mode";
                _predictionEngine?.Dispose();
                _predictionEngine = null;
                Messages.Enqueue("100% Detect Mode Stopped");
                DetectionRunning = false;
            }
        }

        private void StopButton_Click(object sender, RoutedEventArgs e)
        {
            StopAndClose();
        }


        private void StopAndClose()
        {
            _isRunning = false;
            _capture?.Release();
            _predictionEngine?.Dispose();
            Close();
        }

        private void Train_Click(object sender, RoutedEventArgs e)
        {
            if (Messages == null)
            {
                Messages = new ConcurrentQueue<string>();
            }
            Task.Run(() => CreateModel(Messages));
        }
        private void Files_Click(object sender, RoutedEventArgs e)
        {
            string tempfolder = Path.Combine(imagesFolderPathForTraining, LabelInput.Text.Trim());
            Directory.CreateDirectory(tempfolder);
            var psi = new ProcessStartInfo();
            psi.FileName = @"c:\windows\explorer.exe";
            psi.Arguments = tempfolder;
            Process.Start(psi);
        }

        private void LabelInput_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (LabelInput.Text.Contains("-"))
            {
                LabelInput.Text = LabelInput.Text.Replace("-", "_");
            }
            CaptureButton.IsEnabled = (LabelInput.Text.Length > 0);
            FileButton.IsEnabled = (LabelInput.Text.Length > 0);
        }

        private void Window_Closed(object sender, EventArgs e)
        {
            StopAndClose();
        }

        private void RadioButton_Checked(object sender, RoutedEventArgs e)
        {
            useFaceDetection = true;
        }

        private void RefimageCheckbox_Checked(object sender, RoutedEventArgs e)
        {
            useReferenceImage = true;
        }

        private void RefimageCheckbox_UnChecked(object sender, RoutedEventArgs e)
        {
            useReferenceImage = false;
        }

        private void FaceDetectionCheckbox_Unchecked(object sender, RoutedEventArgs e)
        {
            useFaceDetection = false;
        }
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