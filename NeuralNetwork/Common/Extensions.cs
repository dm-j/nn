using System;
using System.Collections.Generic;
using System.Linq;

namespace DMJ.NeuralNetwork.Common
{
    public static class Extensions
    {
        public static IEnumerable<T> Do<T>(this IEnumerable<T> sequence, Action<T> action) =>
            sequence.DoLazy(action).ToArray();

        public static IEnumerable<T> DoLazy<T>(this IEnumerable<T> sequence, Action<T> action)
        {
            foreach (T t in sequence)
            {
                action(t);
                yield return t;
            }
        }

        public static IEnumerable<T> Do<T>(this IEnumerable<T> sequence, Action action) =>
            sequence.DoLazy(action).ToArray();

        public static IEnumerable<T> DoLazy<T>(this IEnumerable<T> sequence, Action action)
        {
            foreach (T t in sequence)
            {
                action();
                yield return t;
            }
        }

        public static IEnumerable<T> Do<T>(this IEnumerable<T> sequence, Action<T, int> action) =>
            sequence.DoLazy(action).ToArray();

        public static IEnumerable<T> DoLazy<T>(this IEnumerable<T> sequence, Action<T, int> action)
        {
            foreach (var t in sequence.Select((o, i) => (o, i)))
            {
                action(t.o, t.i);
                yield return t.o;
            }
        }

        public static IEnumerable<T> Reversed<T>(this IEnumerable<T> sequence)
        {
            T[] asArray = sequence.ToArray();

            for (int index = asArray.Length - 1; index >= 0; index--)
                yield return asArray[index];
        }

        public static int MaxIndex<T>(this IEnumerable<T> sequence) where T : IComparable =>
            sequence.Select((item, index) => (item, index)).MaxBy(o => o.item).index;

        public static int WeightedPickIndex(this IEnumerable<double> sequence)
        {
            double minimum = sequence.Min();
            var results = sequence.Select((weight, index) => (weight: weight - minimum, index));

            double choice = StrongRandom.NextDouble * results.Sum(o => o.weight - minimum);

            double runningWeightTotal = 0d;

            foreach (var (weight, index) in results)
            {
                runningWeightTotal += weight;

                if (runningWeightTotal >= choice)
                    return index;
            }

            return results.MaxBy(o => o.weight).index;
        }

        public static IEnumerable<T> Shuffle<T>(this IEnumerable<T> sequence)
        {
            T[] asArray = sequence.ToArray();
            int length = asArray.Length;
            for (int index = 0; index < length; index++)
            {
                int selection = StrongRandom.Range(index, length);
                T item = asArray[selection];
                asArray[selection] = asArray[index];
                asArray[index] = item;
            }
            return asArray;
        }

        private static T MinMaxBy<T, TKey>(this IEnumerable<T> sequence, Func<T, TKey> selector, int MinMax)
            where TKey : IComparable
        {
            int initCount = sequence.Take(2).Count();
            if (initCount == 0)
                throw new ArgumentException($"Cannot find min or max of an empty enumerable");
            else if (initCount == 1)
                return sequence.First();

            return sequence.Aggregate((result: sequence.First(), value: selector(sequence.First())),
                                      (first, second) => selector(second).CompareTo(first.value) == MinMax
                                                            ? (result: second, value: selector(second))
                                                            : first)
                           .result;
        }

        public static T MinBy<T, TKey>(this IEnumerable<T> sequence, Func<T, TKey> selector)
            where TKey : IComparable =>
            MinMaxBy(sequence, selector, -1);

        public static T MaxBy<T, TKey>(this IEnumerable<T> sequence, Func<T, TKey> selector)
            where TKey : IComparable =>
            MinMaxBy(sequence, selector, 1);

        public static T RandomPick<T>(this IEnumerable<T> sequence)
        {
            var asArray = sequence.ToArray();
            return asArray[StrongRandom.Range(asArray.Length)];
        }

        public static IEnumerable<TItem> Repeat<TItem>(this IEnumerable<TItem> sequence, int repeats)
        {
            for (int i = 0; i < repeats; i++)
            {
                foreach (TItem t in sequence)
                    yield return t;
            }
        }

        public static IEnumerable<TItem> Repeat<TItem>(this IEnumerable<TItem> sequence)
        {
            while (true)
            {
                foreach (TItem t in sequence)
                    yield return t;
            }
        }

        public static IEnumerable<IEnumerable<TOutput>> Batch<TOutput>(this IEnumerable<TOutput> source, int size)
        {
            using (var enumerator = source.GetEnumerator())
                while (enumerator.MoveNext())
                {
                    int i = 0;
                    IEnumerable<TOutput> Batch()
                    {
                        do {
                            yield return enumerator.Current;
                        } while (++i < size && enumerator.MoveNext());
                    };

                    yield return Batch();
                }
        }
    }
}
