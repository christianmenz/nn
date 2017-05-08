package ch.christianmenz.imagefilters;

import java.util.LinkedList;
import java.util.List;
import javax.xml.bind.annotation.XmlRootElement;

/**
 *
 * @author chris
 */
@XmlRootElement
public class NetworkModel {
    
    private List<LayerModel> layers = new LinkedList<>();

    public List<LayerModel> getLayers() {
        return layers;
    }

    public void setLayers(List<LayerModel> layers) {
        this.layers = layers;
    }
    
    
    
}
