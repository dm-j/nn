using DMJ.NeuralNetwork.Common;
using DMJ.NeuralNetwork.Component;
using DMJ.NeuralNetwork.Component.Neurons;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            //First argument will be the path to the images
            //Second argument will be the path to the labels

            int width = 28;
            int height = 28;

            List<Digit> digits = GetLabeledImages(args[0], args[1], width, height).Shuffle().Cast<Digit>().ToList();

            NeuralNet net = NeuralNet.Input(width * height).Layer<ReLU>(20).Layer<LeakyReLU1>(20).Output<Sigmoid>(10);

            IEnumerable<IDataSet> training = digits.Take(digits.Count() - 100);
            IEnumerable<IDataSet> test = digits.Skip(digits.Count() - 100);

            net.TrainByExample(training, epochs: 2, fuzz: 0.2d);

            Console.WriteLine($"\n{test.Where(sample => net.Evaluate(sample.Values).Select((o, i) => (o, i)).MaxBy(o => o.o).i == sample.Targets.MaxIndex()).Count()}/{test.Count()} correct from test set.");
            
            File.WriteAllText(@"C:\NNTrimmed.json", net.ToJson());
        }

        private static IEnumerable<IDataSet> GetLabeledImages(string imagesPath, string labelsPath, int width, int height)
        {
            int elementLength = width * height;
            byte[] buffer = new byte[elementLength];
            using (FileStream imageFileStream = new FileStream(imagesPath, FileMode.Open))
            using (BinaryReader imageBinaryStream = new BinaryReader(imageFileStream))
            using (FileStream labelFileStream = new FileStream(labelsPath, FileMode.Open))
            using (BinaryReader labelBinaryStream = new BinaryReader(labelFileStream))
            {
                imageBinaryStream.ReadBytes(16);
                labelBinaryStream.ReadBytes(8);

                while (imageBinaryStream.Read(buffer, 0, buffer.Length) > 0)
                    yield return new Digit(width, height, buffer, labelBinaryStream.ReadByte()) as IDataSet;
            }
        }

        public class Digit : IDataSet
        {
            private const string asciiGradient = " -=+*#%@@";

            public double[] Values { get; }

            public double[] Targets { get; }
            public readonly int width;
            public readonly int height;
            public readonly int Label;

            public Digit(int width, int height, byte[] pixels, byte label)
            {
                Values = pixels.Select(pixelIntensity => (2d * (Convert.ToDouble(pixelIntensity) / 255d) - 1d)).ToArray();
                Targets = new double [10];
                Targets[label] = 1d;

                this.width = width;
                this.height = height;
                Label = label;
            }

            public override string ToString() =>
                $"{string.Join('\n', Values.Batch(width).Select(line => new string(line.Select(pixel => asciiGradient[(int)(((pixel / 2d) + 0.5d) * (asciiGradient.Length - 1))]).ToArray())))}\nLabel: {Label}";
        }
    }
}
