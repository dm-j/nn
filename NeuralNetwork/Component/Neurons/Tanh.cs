using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using static System.Math;

namespace DMJ.NeuralNetwork.Component.Neurons
{
    public class Tanh : Neuron
    {
        [JsonConstructor]
        private Tanh() : base() { }

        internal Tanh(List<Neuron> inputNeurons, double learningRate, double momentum)
            : base(inputNeurons, learningRate, momentum) { }

        protected override Func<double, double> Activation => HyperbolicTangent;

        protected override Func<double, double> Derivative => x => 1d - Pow(HyperbolicTangent(x), 2);

        private double HyperbolicTangent(double x)
        {
            if (x < -45.0) return -1.0;
            else if (x > 45.0) return 1.0;
            else return Tanh(x);
        }
    }
}
