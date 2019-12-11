// define the routes

var express = require('express'),
    router = express.Router(),
    path = require('path');


router.get('/', function(req, res) {
    res.sendFile(
        'embed.html',
        {root: path.join(__dirname, "../static")});
});


router.get('/embed-distances', function(req, res) {
    res.sendFile(
        'embed-distances.html',
        {root: path.join(__dirname, "../static")});
});


router.get('/embed-many', function(req, res) {
    res.sendFile(
        'embed-many.html',
        {root: path.join(__dirname, "../static")});
});


router.get('/graph', function(req, res) {
    res.sendFile(
        'graph.html',
        {root: path.join(__dirname, "../static")});
});


router.post('/getdata', function(req, res) {
    console.log('hi');
    res.send([proj_points,org_points]);
});


module.exports = router;
