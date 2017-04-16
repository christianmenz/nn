package ch.christianmenz.imagefilters;

import javax.xml.bind.annotation.XmlRootElement;

/**
 *
 * @author Christian
 */
@XmlRootElement
class TrainingData {

    private String trainingInput;

    private String trainingOutput;

    private String testFile;

    public String getTrainingInput() {
        return trainingInput;
    }

    public void setTrainingInput(String trainingInput) {
        this.trainingInput = trainingInput;
    }

    public String getTrainingOutput() {
        return trainingOutput;
    }

    public void setTrainingOutput(String trainingOutput) {
        this.trainingOutput = trainingOutput;
    }

    public String getTestFile() {
        return testFile;
    }

    public void setTestFile(String testFile) {
        this.testFile = testFile;
    }

}
