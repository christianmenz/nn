package ch.christianmenz.imagefilters;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author Christian
 */
public class ImageFilterExample {
    
    private MultiLayerNetwork network;
    
    public static void main(String[] args) throws IOException {
        ImageFilterExample filter = new ImageFilterExample();
        filter.setupNetwork();
        filter.train();
        filter.test();

    }   

    private void setupNetwork() {
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(12)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.SGD)
                .learningRate(0.1d)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(27).nOut(27).activation(Activation.SIGMOID).weightInit(WeightInit.XAVIER).build())
                .layer(1, new DenseLayer.Builder().nIn(27).nOut(8).activation(Activation.SIGMOID).weightInit(WeightInit.XAVIER).build())
                .layer(2, new OutputLayer.Builder().nIn(8).nOut(3).activation(Activation.SIGMOID).weightInit(WeightInit.XAVIER).lossFunction(LossFunctions.LossFunction.L2).build())
                .pretrain(false)
                .backprop(true)
                .build();

        network = new MultiLayerNetwork(configuration);
        network.init();
        network.setListeners(new ScoreIterationListener(1));

    }

    private void train() throws IOException {
        int maxEpoch = 10;

        BufferedImage inputImage = ImageIO.read(new File("cat.png"));
        BufferedImage outputImage = ImageIO.read(new File("cat_heat.png"));
        INDArray input = Nd4j.zeros(27); // reuse
        INDArray output = Nd4j.zeros(3); // reuse

        for (int epoch = 0; epoch < maxEpoch; epoch++) {
            int width = inputImage.getWidth(null);
            int height = inputImage.getHeight(null);

            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    
                    readPixel(inputImage, input, 0, x - 1, y - 1);
                    readPixel(inputImage, input, 1, x, y - 1);
                    readPixel(inputImage, input, 2, x + 1, y - 1);
                    readPixel(inputImage, input, 3, x - 1, y);
                    readPixel(inputImage, input, 4, x, y);
                    readPixel(inputImage, input, 5, x + 1, y);
                    readPixel(inputImage, input, 6, x - 1, y + 1);
                    readPixel(inputImage, input, 7, x, y + 1);
                    readPixel(inputImage, input, 8, x + 1, y + 1);
                    
                    readPixel(outputImage, output, 0, x, y);

                    network.fit(input, output);
                }
            }
        }
    }

    private void readPixel(BufferedImage inputImage, INDArray input, int index, int x, int y) {
        if (x < 0 || y < 0 || x > inputImage.getWidth() - 1 || y > inputImage.getHeight() - 1) {
            input.putScalar(index * 3, 0);
            input.putScalar(index * 3 + 1, 0);
            input.putScalar(index * 3 + 2, 0);
            return;
        }

        int rgb = inputImage.getRGB(x, y);
        Color c = new Color(rgb);
                
        input.putScalar(index * 3, c.getRed() / 255d);
        input.putScalar(index * 3 + 1, c.getGreen() / 255d);
        input.putScalar(index * 3 + 2, c.getBlue() / 255d);        

    }

    private void test() throws IOException {
        BufferedImage inputImage = ImageIO.read(new File("in.png"));
        int height = inputImage.getHeight(null);
        int width = inputImage.getWidth(null);
        
        INDArray input = Nd4j.zeros(27); // reuse

        BufferedImage outputImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                
                readPixel(inputImage, input, 0, x - 1, y - 1);
                readPixel(inputImage, input, 1, x, y - 1);
                readPixel(inputImage, input, 2, x + 1, y - 1);
                readPixel(inputImage, input, 3, x - 1, y);
                readPixel(inputImage, input, 4, x, y);
                readPixel(inputImage, input, 5, x + 1, y);
                readPixel(inputImage, input, 6, x - 1, y + 1);
                readPixel(inputImage, input, 7, x, y + 1);
                readPixel(inputImage, input, 8, x + 1, y + 1);

                INDArray out = network.output(input);

                Color c = new Color((int) (out.getDouble(0) * 255), (int) (out.getDouble(1) * 255), (int) (out.getDouble(2) * 255));                
                outputImage.setRGB(x, y, c.getRGB());

            }

        }
        ImageIO.write(outputImage, "png", new File("test-out.png"));
    }

}
