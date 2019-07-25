using Newtonsoft.Json;
using System;
using System.Collections.Generic;

namespace DMJ.NeuralNetwork.Component.Neurons
{
    public class LeakyReLU : Neuron
    {
        [JsonConstructor]
        private LeakyReLU() : base() { }

        internal LeakyReLU(List<Neuron> inputNeurons, double learningRate, double momentum)
            : base(inputNeurons, learningRate, momentum) { }

        protected override Func<double, double> Activation => x => x > 0d 
                                                                    ? x 
                                                                    : x / 100d;

        protected override Func<double, double> Derivative => x => x > 0d 
                                                                    ? 1d 
                                                                    : 0.01d;
    }
}
