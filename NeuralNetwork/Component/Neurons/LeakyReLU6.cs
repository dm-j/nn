using Newtonsoft.Json;
using System;
using System.Collections.Generic;

namespace DMJ.NeuralNetwork.Component.Neurons
{
    public class LeakyReLU6 : Neuron
    {
        [JsonConstructor]
        private LeakyReLU6() : base() { }

        internal LeakyReLU6(List<Neuron> inputNeurons, double learningRate, double momentum)
            : base(inputNeurons, learningRate, momentum) { }

        protected override Func<double, double> Activation => x => x > 0d
                                                                        ? x > 6d
                                                                            ? x / 50d
                                                                            : 1d
                                                                        : x / 100d;

        protected override Func<double, double> Derivative => x => x > 0d
                                                                       ? x > 6d
                                                                            ? 0.02d
                                                                            : 1d
                                                                       : 0.01d;
    }
}
