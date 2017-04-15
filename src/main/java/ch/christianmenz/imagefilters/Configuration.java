package ch.christianmenz.imagefilters;

import javax.xml.bind.annotation.XmlRootElement;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author Christian
 */
@XmlRootElement
public class Configuration {

    private Activation[] activations;

    private OptimizationAlgorithm[] optimizationAlgos;

    private LossFunctions.LossFunction[] lossFunctions;

    private Updater[] updaters;

    private WeightInit[] weightInits;

    public WeightInit[] getWeightInits() {
        return weightInits;
    }

    public void setWeightInits(WeightInit[] weightInits) {
        this.weightInits = weightInits;
    }

    public Activation[] getActivations() {
        return activations;
    }

    public void setActivations(Activation[] activations) {
        this.activations = activations;
    }

    public OptimizationAlgorithm[] getOptimizationAlgos() {
        return optimizationAlgos;
    }

    public void setOptimizationAlgos(OptimizationAlgorithm[] optimizationAlgos) {
        this.optimizationAlgos = optimizationAlgos;
    }

    public LossFunctions.LossFunction[] getLossFunctions() {
        return lossFunctions;
    }

    public void setLossFunctions(LossFunctions.LossFunction[] lossFunctions) {
        this.lossFunctions = lossFunctions;
    }

    public Updater[] getUpdaters() {
        return updaters;
    }

    public void setUpdaters(Updater[] updaters) {
        this.updaters = updaters;
    }

}
