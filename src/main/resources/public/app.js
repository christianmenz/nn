
angular.module('neuralApp', ['ngFileUpload']).controller('AppCtrl', function ($http, $scope) {

    var that = this;


    that.createNetwork = createNetwork;
    that.startTraining = startTraining;
    that.addFile = addFile;

    that.state = 0;
    that.trainingData = {};
    that.network = {
        learningRate: 0.01,
        activation: 'SIGMOID',
        lossFunction: 'L2',
        updater: 'SGD',
        optimizationAlgo: 'STOCHASTIC_GRADIENT_DESCENT',
        weightInit: 'XAVIER'
    };

    activate();
    ///


    function activate() {
        $http.get('/nn/configuration').then(function (response) {
            that.configuration = response.data;
        })
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
        that.state++;
    }

    function step() {

    }

    function reset() {
        that.state = 0;
    }
})
