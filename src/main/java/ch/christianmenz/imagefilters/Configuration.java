package ch.christianmenz.imagefilters;

import javax.xml.bind.annotation.XmlRootElement;

/**
 *
 * @author Christian
 */
@XmlRootElement
public class Configuration {
    
    private String[] activations;
    
    private String[] optimizationAlgos;
    
    private String[] lossFunctions;
    
    private String[] updaters;
    
}
