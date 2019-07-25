using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using static System.Math;

namespace DMJ.NeuralNetwork.Component.Neurons
{
    public class ReLU6 : Neuron
    {
        [JsonConstructor]
        private ReLU6() : base() { }

        internal ReLU6(List<Neuron> inputNeurons, double learningRate, double momentum)
            : base(inputNeurons, learningRate, momentum) { }

        protected override Func<double, double> Activation => x => Min(Max(0d, x), 6d);

        protected override Func<double, double> Derivative => x => x > 0d 
                                                                       ? x > 6d
                                                                            ? 0d
                                                                            : 1d
                                                                       : 0d;
    }
}
