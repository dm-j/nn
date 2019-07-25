using Newtonsoft.Json;
using DMJ.NeuralNetwork.Common;
using DMJ.NeuralNetwork.Component.Neurons;
using System;
using System.Collections.Generic;
using System.Linq;

namespace DMJ.NeuralNetwork.Component
{
    [JsonObject(IsReference = true)]
    public class NeuralNet
    {
        [JsonProperty] private readonly List<List<Neuron>> network = new List<List<Neuron>>();

        public static NeuralNet FromJson(string json) =>
            JsonConvert.DeserializeObject<NeuralNet>(json, new JsonSerializerSettings() { TypeNameHandling = TypeNameHandling.Auto, NullValueHandling = NullValueHandling.Ignore });

        public string ToJson() =>
            JsonConvert.SerializeObject(this, new JsonSerializerSettings() { TypeNameHandling = TypeNameHandling.Auto, NullValueHandling = NullValueHandling.Ignore, Formatting = Formatting.Indented });

        public static NeuralNetworkBuilder_ Input(int numberOfNeurons, double learningRate = 0.01d, double momentum = 0d) =>
            new NeuralNetworkBuilder_(numberOfNeurons, learningRate, momentum);

        [JsonConstructor]
        private NeuralNet() { }

        internal NeuralNet(List<List<Neuron>> network)
        {
            this.network = network;
        }

        public void TrainByExample(IEnumerable<IDataSet> dataSets, int epochs = 1, double fuzz = 0d) =>
            TrainByExample(dataSets.Select(data => data.Values), dataSets.Select(data => data.Targets), epochs, fuzz);

        public void TrainByExample(IEnumerable<double[]> inputs, IEnumerable<double[]> outputs, int epochs = 1, double fuzz = 0d) =>
            Enumerable.Repeat(0, epochs).Select(_ => inputs.Zip(outputs, (inp, outp) => (inp, outp)))
                                        .SelectMany(set => set.Shuffle())
                                        .Do(sample => TrainByExample(Fuzz(sample.inp, fuzz), sample.outp));

        public void TrainByExample(double[] inputs, double[] outputs)
        {
            Forward(inputs);
            Backward(outputs);
        }

        private double[] Fuzz(double[] values, double fuzz)
        {
            if (fuzz <= 0d)
                return values;

            double[] v = new double[values.Length];
            values.CopyTo(v, 0);
            Enumerable.Range(0, values.Length)
                      .Shuffle()
                      .Take((int)(values.Length * fuzz))
                      .Do(index => v[index] = 2d * StrongRandom.NextDouble - 1d);
            return v;
        }

        public void TrainByExperience(double[] inputs, int choice, double result)
        {
            double[] current = Forward(inputs);
            current[choice] = result;
            Backward(current);
        }

        private double[] Forward(params double[] inputs)
        {
            network.First().OfType<Input>().Do((neuron, index) => neuron.Value = inputs[index]);
            network.Skip(1).Do(layer => layer.Do(neuron => neuron.CalculateValue()));
            return network.Last().Select(neuron => neuron.Value).ToArray();
        }

        private void Backward(params double[] targets)
        {
            network.Last().Do((a, i) => a.CalculateGradient(targets[i]));

            network.Reversed()
                   .Skip(1)
                   .Do(layer => layer.Do(neuron => neuron.CalculateGradient()))
                   .Do(layer => layer.Do(neuron => neuron.UpdateWeights()));

            network.Last().Do(neuron => neuron.UpdateWeights());
        }

        public double[] Evaluate(params double[] inputs) =>
            Forward(inputs);

        public double[] Evaluate(IDataSet inputs) =>
            Forward(inputs.Values);

        public double[] Hallucinate(params double[] outputs)
        {
            network.Last().Do((neuron, index) => neuron.Value = outputs[index]);

            return network.Reversed()
                          .Skip(1)
                          .Do(layer => layer.Do(neuron => neuron.Value = neuron.OutputSynapses.Sum(synapse => synapse.Weight * synapse.OutputNeuron.Value)))
                          .Last()
                          .OfType<Input>()
                          .Select(neuron => neuron.Value)
                          .ToArray();
        }

        public int Kick(double threshold) =>
            Synapses.Where(synapse => Math.Abs(synapse.Weight) <= threshold)
                    .Do(synapse => synapse.Weight = StrongRandom.NormalDouble)
                    .Count();

        public int Trim(double threshold) =>
            Synapses.Where(synapse => Math.Abs(synapse.Weight) <= threshold)
                    .Do(synapse => synapse.Remove())
                    .Count();

        private IEnumerable<Synapse> Synapses =>
            network.SelectMany(neurons => neurons)
                   .SelectMany(neuron => neuron.InputSynapses);
    }
}
