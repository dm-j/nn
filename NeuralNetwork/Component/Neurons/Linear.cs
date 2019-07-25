using Newtonsoft.Json;
using System;
using System.Collections.Generic;

namespace DMJ.NeuralNetwork.Component.Neurons
{
    public class Linear : Neuron
    {
        [JsonConstructor]
        internal Linear() : base() { }

        internal Linear(List<Neuron> inputNeurons, double learningRate, double momentum)
            : base(inputNeurons, learningRate, momentum) { }

        protected override Func<double, double> Activation => x => x;

        protected override Func<double, double> Derivative => x => 1d;
    }
}