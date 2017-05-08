package ch.christianmenz.imagefilters;

import java.util.LinkedList;
import java.util.List;

/**
 *
 * @author chris
 */
public class LayerModel {

    private List<NeuronModel> neurons = new LinkedList<>();

    public List<NeuronModel> getNeurons() {
        return neurons;
    }

    public void setNeurons(List<NeuronModel> neurons) {
        this.neurons = neurons;
    }
}
