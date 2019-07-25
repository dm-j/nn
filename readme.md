##Neural Network code!

This is a test project which creates, trains, saves, and loads a simple and inefficient neural network.

To use the test project, you'll need to download the following files:

http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz (images)

http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz (labels)

You'll need to unzip these files and note where you extract them to, you'll need the paths to these files when you invoke the program.

If you run it from command line, invoke it with the location of the images, then the location of the labels, for example this is how I would invoke it:

`HandwritingRecognition C:\MNIST\train-images.idx3-ubyte C:\MNIST\train-labels.idx1-ubyte`

Alternatively, you can run it in Debug in Visual Studio. Right-click on the HandwritingRecognition project, then under Debug, in `Application arguments`, put the location of the images, then the labels, separated by a space.