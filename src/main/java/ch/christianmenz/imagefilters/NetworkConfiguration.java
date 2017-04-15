package ch.christianmenz.imagefilters;

import javax.xml.bind.annotation.XmlRootElement;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author Christian
 */
@XmlRootElement
public class NetworkConfiguration {

    private int radius;

    private int[] layers;

    private double learningRate;

    private Activation activation;

    private OptimizationAlgorithm optimizationAlgo;

    private Updater updater;

    private LossFunctions.LossFunction lossFunction;

    private double momentum;

    public int getRadius() {
        return radius;
    }

    public void setRadius(int radius) {
        this.radius = radius;
    }

    public int[] getLayers() {
        return layers;
    }

    public void setLayers(int[] layers) {
        this.layers = layers;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public Activation getActivation() {
        return activation;
    }

    public void setActivation(Activation activation) {
        this.activation = activation;
    }

    public OptimizationAlgorithm getOptimizationAlgo() {
        return optimizationAlgo;
    }

    public void setOptimizationAlgo(OptimizationAlgorithm optimizationAlgo) {
        this.optimizationAlgo = optimizationAlgo;
    }

    public Updater getUpdater() {
        return updater;
    }

    public void setUpdater(Updater updater) {
        this.updater = updater;
    }

    public LossFunctions.LossFunction getLossFunction() {
        return lossFunction;
    }

    public void setLossFunction(LossFunctions.LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }

    public double getMomentum() {
        return momentum;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

}
