using Newtonsoft.Json;
using System;
using System.Collections.Generic;

namespace DMJ.NeuralNetwork.Component.Neurons
{
    public class LeakyReLU1 : Neuron
    {
        [JsonConstructor]
        private LeakyReLU1() : base() { }

        internal LeakyReLU1(List<Neuron> inputNeurons, double learningRate, double momentum)
            : base(inputNeurons, learningRate, momentum) { }

        protected override Func<double, double> Activation => x => x > 0d
                                                                        ? x > 1d
                                                                            ? x / 50d
                                                                            : 1d
                                                                        : x / 100d;

        protected override Func<double, double> Derivative => x => x > 0d
                                                                       ? x > 1d
                                                                            ? 0.02d
                                                                            : 1d
                                                                       : 0.01d;
    }
}
