using Microsoft.ML.Data;

namespace PrecipitationMLNet
{
    internal class PrecipitationPrediction
    {
        [ColumnName("Score")]
        public float PredictedPrecipitation { get; set; }
    }
}