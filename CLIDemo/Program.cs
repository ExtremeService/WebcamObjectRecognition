using ImageClassification;
using Microsoft.ML;
using static ImageClassification.MLModel;
namespace CLIDemo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            CreateDirectories();
            MLModel.outputMlNetModelFileName = "MLModel.zip";


            //var singleMarco = "";//TakeSinglePicturesForPrediction("Marco");
            LoadModelandPredict("C:\\Temp\\MLTraining\\assets\\inputs\\images\\marco\\638801515384410111.jpg");




            TakePictures("marco",100);
            CreateModel();



            var singleMarco = TakeSinglePicturesForPrediction("Marco");
            LoadModelandPredict(singleMarco[0]);


        }
    }
}
