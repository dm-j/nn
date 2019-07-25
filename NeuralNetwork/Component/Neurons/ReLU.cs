using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using static System.Math;

namespace DMJ.NeuralNetwork.Component.Neurons
{
    public class ReLU : Neuron
    {
        [JsonConstructor]
        private ReLU() : base() { }

        internal ReLU(List<Neuron> inputNeurons, double learningRate, double momentum)
            : base(inputNeurons, learningRate, momentum) { }

        protected override Func<double, double> Activation => x => Max(0d, x);

        protected override Func<double, double> Derivative => x => x > 0d ? 1d : 0d;
    }
}
