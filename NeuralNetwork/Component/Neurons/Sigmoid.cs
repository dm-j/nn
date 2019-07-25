using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using static System.Math;

namespace DMJ.NeuralNetwork.Component.Neurons
{
    public class Sigmoid : Neuron
    {
        [JsonConstructor]
        private Sigmoid() : base() { }

        internal Sigmoid(List<Neuron> inputNeurons, double learningRate, double momentum)
            : base(inputNeurons, learningRate, momentum) { }

        protected override Func<double, double> Activation => x => x < -45.0 ? 0.0 : x > 45.0 ? 1.0 : 1.0 / (1.0 + Exp(-x));

        protected override Func<double, double> Derivative => x => x * (1 - x);
    }
}
