var requirejs = require('requirejs');

requirejs.config({
    nodeRequire: require,
    paths: {
        pointnd: "static/js/models/pointnd",
        distmat: "static/js/models/distancematrix"
    }
});

requirejs(['jquery', 'express', 'fs', 'pointnd', 'distmat'],
function($, express, fs, PointND, DistanceMatrix) {

    var app = express();

    app.use('/', require('./routes/index'));


    get_cluster_data = function get_cluster_data(filename) {

        data = fs.readFileSync(filename);

        // parse the string (split into lines)
        data_string= data.toString();
        linesplit = data_string.split('\n');
        //console.log(data_string);

        points = [];

        // first line is the header
        for(var i = 0; i < linesplit.length-1; i++) {
        //for(var i = 0; i < 5000; i++) {
        //for(var i = 0; i < 100; i++) {
            if(linesplit[i] === undefined)
                console.log("hi");
            //console.log('   "' + linesplit[i] + '"');
            tabsplit = linesplit[i].split(',');
            //console.log(tabsplit);

            point = [];
            for(var j = 0; j < tabsplit.length-3; j++) {
                point.push(parseFloat(tabsplit[j]));
            }
            cluster_id = parseFloat(tabsplit[tabsplit.length-3]);
            media_src = tabsplit[tabsplit.length-2];
            media_src2 = tabsplit[tabsplit.length-1];
            points.push(new PointND(i, point, cluster_id, media_src, media_src2));

        }

        return points;
    };


    //proj_points = get_cluster_data('static/proj_data.csv');
    //org_points = get_cluster_data('static/org_data.csv');
    //org_points = get_cluster_data('/groups/branson/bransonlab/kwaki/cache/mnistmulticriterion/embeddings/embedding_org.csv');
    //proj_points = get_cluster_data('/groups/branson/bransonlab/kwaki/cache/mnistmulticriterion/embeddings/embedding_00360000.csv');
    //proj_points = get_cluster_data('/groups/branson/bransonlab/kwaki/cache/multicriterion/embeddings/embedding_org.csv');
    //org_points = get_cluster_data('static/embeddings/embedding_test.csv');
    //proj_points = get_cluster_data('static/embeddings/embedding_test_proj.csv');
    org_points = get_cluster_data('static/orgembeddings/embedding_org.csv');
    proj_points = get_cluster_data('static/orgembeddings/embedding_org_proj.csv');

    app.use(express.static(__dirname + '/static'));
    app.listen(4000, function() {
        console.log('beep boop');
    });
});
