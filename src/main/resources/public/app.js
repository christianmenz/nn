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

            document.getElementById('networkPaper').innerHTML = '';
            var paper = new Raphael(document.getElementById('networkPaper'));

            var xMargin = 1200;
            var yMargin = 60;
            var circleRadius = 60;

            var maxNeurons = 0;

            // find "highest" layer 
            for (var l = 0; l < that.networkModel.layers.length; l++) {
                let n = that.networkModel.layers[l].neurons.length;
                maxNeurons = n > maxNeurons ? n : maxNeurons;
            }

            // find dimensions
            let height = maxNeurons * (circleRadius * 2 + yMargin) + yMargin;
            let width = that.networkModel.layers.length * (circleRadius * 2 + xMargin) + xMargin;
            paper.setViewBox(0, 0, width, height, true);
            //paper.canvas.setAttribute('preserveAspectRatio', 'none');                       


            for (var l = 0; l < that.networkModel.layers.length; l++) {
                let layer = that.networkModel.layers[l];
                let layerHeight = layer.neurons.length * (circleRadius * 2 + yMargin) + yMargin;
                let yOffset = (height - layerHeight) / 2;
                let x = l * (circleRadius * 2 + xMargin) + xMargin;

                for (let n = 0; n < layer.neurons.length; n++) {
                    let y = n * (circleRadius * 2 + yMargin) + yOffset + yMargin;

                    // for each we need a connection to the next layer...
                    if (l < that.networkModel.layers.length - 1) {
                        let nextLayer = that.networkModel.layers[l + 1];
                        let nextLayerHeight = nextLayer.neurons.length * (circleRadius * 2 + yMargin) + yMargin;
                        let nextLayerYOffset = (height - nextLayerHeight) / 2;
                        let nextLayerX = (l + 1) * (circleRadius * 2 + xMargin) + xMargin;

                        for (let nextN = 0; nextN < nextLayer.neurons.length; nextN++) {
                            let nextY = nextN * (circleRadius * 2 + yMargin) + nextLayerYOffset + yMargin;
                            let p = paper.path('M' + x + ' ' + y + 'L' + nextLayerX + ' ' + nextY);
                            p.attr('stroke', nextLayer.neurons[nextN].weights[n] > 0 ? 'green' : 'red');
                            p.attr('stroke-width', Math.abs(nextLayer.neurons[nextN].weights[n]) * 10);
                        }
                    }

                    let c = paper.circle(x, y, circleRadius);
                    c.attr("stroke", "black");
                    c.attr('stroke-width', '5');
                    c.attr('fill', 'lightgray');
                }
            }
        })
    }
})
