using Newtonsoft.Json;
using DMJ.NeuralNetwork.Common;
using System;

namespace DMJ.NeuralNetwork.Component
{
    [JsonObject(IsReference = true)]
    public class Synapse
    {
        [JsonProperty] public Neuron InputNeuron { get; set; }
        [JsonProperty] public Neuron OutputNeuron { get; set; }
        [JsonProperty] public double Weight { get; set; }
        [JsonIgnore] public double WeightDelta { get; set; }
        
        [JsonConstructor]
        public Synapse() { }

        public Synapse(Neuron inputNeuron, Neuron outputNeuron)
        {
            InputNeuron = inputNeuron;
            OutputNeuron = outputNeuron;
            Weight = StrongRandom.NormalDouble;
        }

        public void Update()
        {
            var prevDelta = WeightDelta;
            WeightDelta = OutputNeuron.LearningRate * OutputNeuron.Gradient * InputNeuron.Value;
            Weight += ((1d - OutputNeuron.Momentum) * WeightDelta) + (OutputNeuron.Momentum * prevDelta);
        }

        public void Remove()
        {
            InputNeuron?.OutputSynapses.Remove(this);
            OutputNeuron?.InputSynapses.Remove(this);
        }
    }
}
