
angular.module('neuralApp', []).controller('AppCtrl', function ($http) {

    var that = this;

    that.network = {
        learningRate: 0.01,
        activation: 'SIGMOID',
        lossFunction: 'L2',
        updater: 'SGD',
        optimizationAlgo: 'STOCHASTIC_GRADIENT_DESCENT'
    };



    activate();
    ///


    function activate() {
        $http.get('/nn/configuration').then(function (response) {
            that.configuration = response.data;
        })
    }
})
