package ch.christianmenz.imagefilters;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import javax.imageio.ImageIO;
import org.apache.commons.codec.binary.Base64;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
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

    private int batchSize;
    
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

    private BufferedImage getImageFromDataUrl(String dataUrl) throws IOException {
        String encodingPrefix = "base64,";
        int contentStartIndex = dataUrl.indexOf(encodingPrefix) + encodingPrefix.length();
        byte[] imageData = Base64.decodeBase64(dataUrl.substring(contentStartIndex));
        return ImageIO.read(new ByteArrayInputStream(imageData));
    }

    @RequestMapping(path = "provideTrainingData", method = RequestMethod.POST)
    public void provideTrainingData(@RequestBody TrainingData trainingData) throws IOException {
        inputImage = getImageFromDataUrl(trainingData.getTrainingInput());
        outputImage = getImageFromDataUrl(trainingData.getTrainingOutput());
        testOutputImage = getImageFromDataUrl(trainingData.getTestFile());
        testImage = getImageFromDataUrl(trainingData.getTestFile());
    }

    @RequestMapping(path = "configuration", method = RequestMethod.GET)
    public ResponseEntity<Configuration> configuration() {
        Configuration config = new Configuration();
        config.setActivations(
                Activation.SIGMOID,
                Activation.RELU, 
                Activation.SOFTMAX, 
                Activation.TANH);
        config.setLossFunctions(
                LossFunction.COSINE_PROXIMITY,
                LossFunction.L1,
                LossFunction.L2,
                LossFunction.MEAN_ABSOLUTE_ERROR,
                LossFunction.MSE,
                LossFunction.NEGATIVELOGLIKELIHOOD);
        config.setOptimizationAlgos(
                OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT, 
                OptimizationAlgorithm.LINE_GRADIENT_DESCENT);
        config.setWeightInits(
                WeightInit.XAVIER,
                WeightInit.RELU, 
                WeightInit.DISTRIBUTION, 
                WeightInit.ZERO);
        config.setUpdaters(
                Updater.NONE, 
                Updater.SGD, 
                Updater.ADAM, 
                Updater.ADAGRAD, 
                Updater.ADADELTA);
        return ResponseEntity.ok(config);
    }

    @RequestMapping(path = "trainingStep", method = RequestMethod.POST)
    public NetworkModel trainingStep() throws IOException {

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
        // update network infos      
        NetworkModel model = new NetworkModel();

        for (Layer l : network.getLayers()) {
            LayerModel layerModel = new LayerModel();
            INDArray weights = l.getParam("W");
            INDArray biases = l.getParam("b");
            for (int i = 0; i < weights.columns(); i++) {
                NeuronModel n = new NeuronModel();
                n.setWeights(weights.getColumn(i).dup().data().asDouble());
                n.setBias(biases.getDouble(i));
                layerModel.getNeurons().add(n);

            }
            model.getLayers().add(layerModel);
        }
        return model;
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
