angular.module('neuralApp', ['ngFileUpload']).controller('AppCtrl', function ($http, $scope) {

    var that = this;

    that.createNetwork = createNetwork;
    that.startTraining = startTraining;
    that.trainingStep = trainingStep;
    that.startOver = startOver;
    that.addFile = addFile;

    that.state = 0;
    that.training = false;
    that.iteration = 0;
    that.imagesSelcted = false;
    that.trainingData = {};
    that.network = {
        learningRate: 0.1,
        batchSize: 500,
        activation: 'SIGMOID',
        lossFunction: 'L2',
        updater: 'SGD',
        optimizationAlgo: 'STOCHASTIC_GRADIENT_DESCENT',
        weightInit: 'XAVIER',
        momentum: 0.1
    };

    activate();
    ///


    function activate() {
        $http.get('/nn/configuration').then(function (response) {
            that.configuration = response.data;
        })
    }

    function startOver() {
        that.state = 0;
    }

    function addFile($file, fileType) {

        var reader = new FileReader();

        if (!$file) {
            return;
        }

        reader.onload = function (e) {
            var fileContent;
            fileContent = e.target.result;
            that.trainingData[fileType] = fileContent;

            if (that.trainingData.trainingInput && that.trainingData.trainingOutput && that.trainingData.testFile) {
                that.imagesSelcted = true;
            }

            $scope.$apply();
        };
        reader.readAsDataURL($file);

    }

    function createNetwork() {
        $http.post('/nn/configure', that.network).then(function (response) {
            that.state++;
        })
    }

    function startTraining() {
        $http.post('/nn/provideTrainingData', that.trainingData).then(function (response) {
            that.state++;
        })
    }


    function trainingStep() {
        that.training = true;
        $http.post('/nn/trainingStep').then(function (response) {
            that.iteration++;
            that.training = false;
            that.networkModel = response.data;

            var paper = new Raphael(document.getElementById('networkPaper'), 800, 800);
            paper.setViewBox(0, 0, 2000, 2000, true);

            var xOffset = 50;
            var yOffset = 50;

            for (var l = 0; l < that.networkModel.layers.length; l++) {
                var layer = that.networkModel.layers[l];
                var x = l * 300 + xOffset;

                for (var n = 0; n < layer.neurons.length; n++) {
                    paper.circle(x, n * (60 + yOffset), 30);

                }
            }




        })
    }
})
