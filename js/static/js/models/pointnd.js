define([], function() {

    var PointND = function(idx, coords, cluster, media_src, media_src2){
        this.idx = idx;
        this.vector = coords;
        this.media_src = media_src;
        this.media_src2 = media_src2;
        //this.point_type = point_type;
        this.cluster = cluster;
    };

    PointND.prototype.print = function() {
        console.log(this);
    };

    // get the euclidean distace
    PointND.eucliddist = function dist(p1, p2) {
        var sqdist = 0;
        for(var i = 0; i < p1.vector.length; i++) {
            sqdist = Math.pow(p1.vector[i] - p2.vector[i], 2) + sqdist;
        }
        return Math.pow(sqdist, 0.5);
    };

    // build getters & setters
    return PointND;
});

