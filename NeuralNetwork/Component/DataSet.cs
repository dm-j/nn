namespace DMJ.NeuralNetwork.Component
{
    public interface IDataSet
    {
        double[] Values { get; }
        double[] Targets { get; }
    }
}
