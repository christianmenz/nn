package ch.christianmenz.imagefilters;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import javax.imageio.ImageIO;
import org.apache.commons.codec.binary.Base64;
import org.deeplearning4j.eval.Evaluation;
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
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.context.annotation.SessionScope;
import org.springframework.web.bind.annotation.RequestBody;

/**
 *
 * @author Christian
 */
@RestController
@SessionScope
@RequestMapping("/nn")
public class NeuralNetwork {

    private MultiLayerNetwork network;

    private int epoch;

    private int iteration;

    private int batchSize;

    private Evaluation evaluation;

    private BufferedImage inputImage;

    private BufferedImage outputImage;

    private BufferedImage testImage;

    private BufferedImage testOutputImage;

    private int x;

    private int y;

    @RequestMapping(path = "configure", method = RequestMethod.POST)
    public void configure(@RequestBody NetworkConfiguration networkConfiguration) {
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(12)
                .optimizationAlgo(networkConfiguration.getOptimizationAlgo())
                .updater(networkConfiguration.getUpdater())
                .learningRate(networkConfiguration.getLearningRate())                
                .momentum(networkConfiguration.getMomentum())
                .list()
                .layer(0, new DenseLayer.Builder().nIn(27).nOut(27).activation(networkConfiguration.getActivation()).weightInit(networkConfiguration.getWeightInit()).build())
                .layer(1, new DenseLayer.Builder().nIn(27).nOut(8).activation(networkConfiguration.getActivation()).weightInit(networkConfiguration.getWeightInit()).build())
                .layer(2, new OutputLayer.Builder().nIn(8).nOut(3).activation(networkConfiguration.getActivation()).weightInit(networkConfiguration.getWeightInit()).lossFunction(networkConfiguration.getLossFunction()).build())
                .pretrain(false)
                .backprop(true)
                .build();

        network = new MultiLayerNetwork(configuration);
        network.init();
        network.setListeners(new ScoreIterationListener(1000));
        batchSize = networkConfiguration.getBatchSize();
    }

    @RequestMapping(path = "provideTrainingData", method = RequestMethod.POST)
    public void provideTrainingData(@RequestBody TrainingData trainingData) throws IOException {
        String encodingPrefix = "data:image/png;base64,";

        inputImage = ImageIO.read(new ByteArrayInputStream(Base64.decodeBase64(trainingData.getTrainingInput().substring(encodingPrefix.length()))));
        outputImage = ImageIO.read(new ByteArrayInputStream(Base64.decodeBase64(trainingData.getTrainingOutput().substring(encodingPrefix.length()))));
        testOutputImage = ImageIO.read(new ByteArrayInputStream(Base64.decodeBase64(trainingData.getTestFile().substring(encodingPrefix.length()))));
        testImage = ImageIO.read(new ByteArrayInputStream(Base64.decodeBase64(trainingData.getTestFile().substring(encodingPrefix.length()))));
    }

    @RequestMapping(path = "configuration", method = RequestMethod.GET)
    public ResponseEntity<Configuration> configuration() {
        Configuration config = new Configuration();
        config.setActivations(Activation.values());
        config.setLossFunctions(LossFunction.values());
        config.setOptimizationAlgos(OptimizationAlgorithm.values());
        config.setWeightInits(WeightInit.values());
        config.setUpdaters(Updater.values());        
        return ResponseEntity.ok(config);
    }

    @RequestMapping(path = "trainingStep", method = RequestMethod.POST)
    public void trainingStep() throws IOException {

        INDArray input = Nd4j.zeros(27); // reuse
        INDArray output = Nd4j.zeros(3); // reuse

        int width = inputImage.getWidth(null);
        int height = inputImage.getHeight(null);

        int iterationCounter = 0;

        loop:
        for (; x < width; x++) {
            for (; y < height; y++) {

                iterationCounter++;

                if (iterationCounter >= batchSize) {
                    break loop;
                }

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
                
                network.fit(input, output); // really fit each single pixel?
            }
            y = 0; // reset the y counter when I get here..
        }

        if (x >= width) {
            x = 0;
        }

        if (y >= height) {
            y = 0;
        }

        test();
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

    @RequestMapping(value = "/testOutput.png", method = RequestMethod.GET, produces = MediaType.IMAGE_PNG_VALUE)
    public ResponseEntity<byte[]> getTestoutput() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(testOutputImage, "png", baos);
        return ResponseEntity.ok(baos.toByteArray());
    }

    private void test() throws IOException {
        int height = testImage.getHeight(null);
        int width = testImage.getWidth(null);

        INDArray allInputs = Nd4j.zeros(height * width, 27); // reuse

        testOutputImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        int co = 0;

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {                

                Color testColor = new Color(testImage.getRGB(x, y), true);
                if (testColor.getAlpha() == 0) {
                    testOutputImage.setRGB(x, y, testColor.getRGB()); // alpha??
                    //continue;
                }

                INDArray input = Nd4j.zeros(27);

                readPixel(testImage, input, 0, x - 1, y - 1);
                readPixel(testImage, input, 1, x, y - 1);
                readPixel(testImage, input, 2, x + 1, y - 1);
                readPixel(testImage, input, 3, x - 1, y);
                readPixel(testImage, input, 4, x, y);
                readPixel(testImage, input, 5, x + 1, y);
                readPixel(testImage, input, 6, x - 1, y + 1);
                readPixel(testImage, input, 7, x, y + 1);
                readPixel(testImage, input, 8, x + 1, y + 1);

                allInputs.putRow(co, input);
                co++;

            }
        }
        INDArray out = network.output(allInputs);

        co = 0;
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {               
                INDArray row = out.getRow(co);
                Color c = new Color((int) (row.getDouble(0) * 255), (int) (row.getDouble(1) * 255), (int) (row.getDouble(2) * 255));
                testOutputImage.setRGB(x, y, c.getRGB());               
                co++;
            }
        }
    }

}
