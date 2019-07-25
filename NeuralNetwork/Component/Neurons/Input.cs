using Newtonsoft.Json;

namespace DMJ.NeuralNetwork.Component.Neurons
{
    public class Input : Linear
    {
        [JsonConstructor]
        internal Input() : base() { }

        public override void CalculateValue() { }
    }
}
