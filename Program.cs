using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PrecipitationMLNet
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            //Load data
            var trainData = context.Data.LoadFromTextFile<FarmWeatherData>("./Farm_Weather_Data.csv", hasHeader: true, separatorChar: ',');

            var testTrainSplit = context.Data.TrainTestSplit(trainData, testFraction: 0.2);

           
            //Build model
            var pipeline = context.Transforms.Concatenate("Features", new string[] { "MaxT", "MinT", "WindSpeed", "Humidity" })
                .Append(context.Regression.Trainers.LbfgsPoissonRegression());

            var model = pipeline.Fit(testTrainSplit.TrainSet);

            
            //Evaluate
            var predictions = model.Transform(testTrainSplit.TestSet);

            var metrics = context.Regression.Evaluate(predictions);

            Console.WriteLine($"R^2 - { metrics.RSquared}");

            //Predict
            var newData = new FarmWeatherData
            {
               MaxT = 23.0f,
            MinT = 12.0f,
            WindSpeed = 16.0f,
            Humidity = 40.0f 

            };

            var predictionFunc = context.Model.CreatePredictionEngine<FarmWeatherData, PrecipitationPrediction>(model);

            var prediction = predictionFunc.Predict(newData);

            Console.WriteLine($"Prediction - {prediction.PredictedPrecipitation}");

            Console.ReadLine();



        }
    }
}
