using Newtonsoft.Json;
using DMJ.NeuralNetwork.Common;
using System;
using System.Collections.Generic;
using System.Linq;

namespace DMJ.NeuralNetwork.Component
{
    [JsonObject(IsReference = true)]
    public abstract class Neuron
    {
        [JsonProperty] internal List<Synapse> InputSynapses { get; set; } = new List<Synapse>();
        [JsonProperty] internal List<Synapse> OutputSynapses { get; set; } = new List<Synapse>();
        [JsonProperty] internal double LearningRate { get; set; }
        [JsonProperty] internal double Momentum { get; set; }
        [JsonIgnore] public double Gradient { get; set; }
        [JsonIgnore] public virtual double Value { get; set; }
        [JsonIgnore] protected abstract Func<double, double> Activation { get; }
        [JsonIgnore] protected abstract Func<double, double> Derivative { get; }

        [JsonConstructor]
        protected Neuron()
        { }

        public Neuron(List<Neuron> inputNeurons, double learningRate, double momentum)
        {
            LearningRate = learningRate;
            Momentum = momentum;
            inputNeurons.Do(neuron => AddSynapse(neuron));
        }

        private void AddSynapse(Neuron inputNeuron)
        {
            var synapse = new Synapse(inputNeuron, this);
            inputNeuron.OutputSynapses.Add(synapse);
            InputSynapses.Add(synapse);
        }

        public virtual void CalculateValue() =>
            Value = Activation(InputSynapses.Sum(synapse => synapse.Weight * synapse.InputNeuron.Value));

        public double CalculateGradient(double target) =>
            Gradient = (target - Value) * Derivative(Value);

        public double CalculateGradient() =>
            Gradient = OutputSynapses.Sum(synapse => synapse.OutputNeuron.Gradient * synapse.Weight) * Derivative(Value);

        public void UpdateWeights() =>
            InputSynapses.Do(synapse => synapse.Update());
    }
}
