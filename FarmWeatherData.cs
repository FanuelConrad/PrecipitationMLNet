using Microsoft.ML.Data;

namespace PrecipitationMLNet
{
    public class FarmWeatherData
    {
        [LoadColumn(2)]
        public float MaxT { get; set; }

        [LoadColumn(3)]
        public float MinT { get; set; }

        [LoadColumn(4)]
        public float WindSpeed { get; set; }

        [LoadColumn(5)]
        public float Humidity { get; set; }

        [LoadColumn(6),ColumnName("Label")]
        public float Precipitation { get; set; }
    }
}