define([], function() {

    var Point = function(id, x, y, cluster, point_type, media_src){
        this.id = id;
        this.x = x;
        this.y = y;
        this.media_src = media_src;
        this.point_type = point_type;
        this.cluster = cluster;
    };

    Point.prototype.print = function() {
        console.log(this);
    };

    Point.eucliddist = function dist(p1, p2) {
        var sqdist = Math.pow(p1.x - p2.x,2) + Math.pow(p1.y - p2.y,2);
        return Math.pow(sqdist, 0.5);
    };

    // build getters & setters
    return Point;
});
