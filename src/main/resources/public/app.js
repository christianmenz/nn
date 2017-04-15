
angular.module('neuralApp', []).controller('AppCtrl', function ($http) {

    var that = this;

    that.state = 0;

    this.createNetwork = createNetwork;

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

    function createNetwork() {
        that.state++;
    }

    function setTrainingData() {
        that.state++;
    }

    function step() {

    }

    function reset() {
        that.state = 0;
    }
})
