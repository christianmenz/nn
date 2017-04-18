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
        })
    }
})
