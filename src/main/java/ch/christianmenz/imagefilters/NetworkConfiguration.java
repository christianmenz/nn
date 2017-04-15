package ch.christianmenz.imagefilters;

import javax.xml.bind.annotation.XmlRootElement;
import org.nd4j.linalg.activations.Activation;

/**
 *
 * @author Christian
 */
@XmlRootElement
public class NetworkConfiguration {

    private int radius;

    private int[] layers;

    private double learningRate;

    private String activation;

    private String optimizationAlgo;

    private String updater;

    private String lossFunction;

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

    public String getActivation() {
        return activation;
    }

    public void setActivation(String activation) {
        this.activation = activation;
    }

    public String getOptimizationAlgo() {
        return optimizationAlgo;
    }

    public void setOptimizationAlgo(String optimizationAlgo) {
        this.optimizationAlgo = optimizationAlgo;
    }

    public String getUpdater() {
        return updater;
    }

    public void setUpdater(String updater) {
        this.updater = updater;
    }

    public String getLossFunction() {
        return lossFunction;
    }

    public void setLossFunction(String lossFunction) {
        this.lossFunction = lossFunction;
    }

    public double getMomentum() {
        return momentum;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

}
