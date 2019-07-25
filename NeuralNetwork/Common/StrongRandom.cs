using System;
using System.Security.Cryptography;

namespace DMJ.NeuralNetwork.Common
{
    public class StrongRandom
    {
        private readonly static RNGCryptoServiceProvider provider = new RNGCryptoServiceProvider();

        public static double NextDouble
        {
            get
            {
                byte[] buffer = new byte[8];
                provider.GetBytes(buffer);
                var ul = BitConverter.ToUInt64(buffer, 0) / (1 << 11);
                return ul / (double)(1UL << 53);
            }
        }

        public static double NormalDouble =>
            NextDouble - NextDouble;

        public static double Range(double max) =>
            NextDouble * max;

        public static double Range(double min, double max) =>
            Range(max - min) + min;

        public static int Range(int max) =>
            (int)Range((double)max);

        public static int Range(int min, int max) =>
            Range(max - min) + min;
    }
}
