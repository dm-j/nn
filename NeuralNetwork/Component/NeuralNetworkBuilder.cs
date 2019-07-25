using DMJ.NeuralNetwork.Component.Neurons;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace DMJ.NeuralNetwork.Component
{
    public class NeuralNetworkBuilder
    {
        private readonly List<List<Neuron>> network;
        private readonly double learningRate;
        private readonly double momentum;

        internal NeuralNetworkBuilder(List<List<Neuron>> network, double learningRate, double momentum)
        {
            this.learningRate = learningRate;
            this.momentum = momentum;
            this.network = network;
        }

        private Neuron CreateNeuron<TNeuron>() where TNeuron : Neuron =>
            Activator.CreateInstance(typeof(TNeuron), BindingFlags.NonPublic | BindingFlags.Instance, null, new object[] { network.Last(), learningRate, momentum }, null) as Neuron;

        public NeuralNetworkBuilder Layer<TNeuron>(int neurons) where TNeuron : Neuron
        {
            var hiddenLayer = Enumerable.Range(1, neurons).Select(_ => CreateNeuron<TNeuron>()).ToList();
            hiddenLayer.Add(new Bias());
            network.Add(hiddenLayer);
            return this;
        }

        public NeuralNet Output<TNeuron>(int neurons) where TNeuron : Neuron
        {
            var outputLayer = Enumerable.Range(1, neurons).Select(_ => CreateNeuron<TNeuron>()).ToList();
            network.Add(outputLayer);
            return new NeuralNet(network);
        }
    }

    public class NeuralNetworkBuilder_
    {
        private readonly double learningRate;
        private readonly double momentum;
        private readonly List<List<Neuron>> network = new List<List<Neuron>>();

        internal NeuralNetworkBuilder_(int neurons, double learningRate, double momentum)
        {
            this.learningRate = learningRate;
            this.momentum = momentum;
            var inputLayer = Enumerable.Range(1, neurons).Select(_ => new Input() as Neuron).ToList();
            inputLayer.Add(new Bias());
            network.Add(inputLayer);
        }

        public NeuralNetworkBuilder Layer<TNeuron>(int neurons) where TNeuron : Neuron
        {
            var hiddenLayer = Enumerable.Range(1, neurons).Select(_ => Activator.CreateInstance(typeof(TNeuron), BindingFlags.NonPublic | BindingFlags.Instance, null, new object[] { network.Last(), learningRate, momentum }, null) as Neuron).ToList();
            hiddenLayer.Add(new Bias());
            network.Add(hiddenLayer);
            return new NeuralNetworkBuilder(network, learningRate, momentum);
        }
    }
}
