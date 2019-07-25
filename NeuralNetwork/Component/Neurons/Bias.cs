using Newtonsoft.Json;
using System;

namespace DMJ.NeuralNetwork.Component.Neurons
{
    public class Bias : Linear
    {
        [JsonConstructor]
        internal Bias()
        { }

        public override double Value => 1d;
    }
}
