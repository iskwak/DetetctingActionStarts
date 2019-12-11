define(["pointnd"], function(PointND) {

    var DistanceMatrix = function(points) {
        this.size = points.length;
        //this.distances = new Array(points.length*points.length);
        this.distances = new Array(points.length);
        this.sum = 0;
        this.sumsq = 0;
        this.max = -1;

        for(var i = 0; i < points.length; i++) {
            this.distances[i] = new Array(points.length);

            for(var j = 0; j < points.length; j++) {
                //var idx = this.size*i + j;
                //this.distances[idx] = PointND.eucliddist(
                //    points[i], points[j]);
                this.distances[i][j] = PointND.eucliddist(
                    points[i], points[j]);
                this.sum = this.sum + this.distances[i][j];
                this.sumsq = this.sumsq + this.distances[i][j]*this.distances[i][j];

                if(this.distances[i][j] > this.max) {
                    this.max = this.distances[i][j];
                }
            }
        }

        var num_el = points.length*points.length;
        this.mean = this.sum / num_el;
        this.std = (this.sumsq - (this.sum*this.sum)/num_el)/(num_el - 1);
        this.std = Math.sqrt(this.std);

        console.log('mean: ' + this.mean);
        console.log('std: ' + this.std);
        console.log('max: ' + this.max);


        this.minus = function minus(dist_mat) {
            this.sum = 0;
            this.sumsq = 0;
            this.max = -1;
            
            for(var i = 0; i < this.size; i++) {
                for(var j = 0; j < this.size; j++) {
                    this.distances[i][j] =
                        Math.abs(this.distances[i][j] - dist_mat.distances[i][j]);

                    this.sum = this.sum + this.distances[i][j];
                    this.sumsq = this.sumsq + this.distances[i][j]*this.distances[i][j];

                    if(this.distances[i][j] > this.max) {
                        this.max = this.distances[i][j];
                    }
                    
                    //console.log(this.distances[i][j] + ' ');
                }
                //console.log('\n');
            }

            var num_el = points.length*points.length;
            this.mean = this.sum / num_el;
            this.std = (this.sumsq - (this.sum*this.sum)/num_el)/(num_el - 1);
            this.std = Math.sqrt(this.std);

            console.log('mean: ' + this.mean);
            console.log('std: ' + this.std);
            console.log('max: ' + this.max);
        };
        //this.at = function at(i,j) {
        //    var idx = this.size*i + j;
        //    return this.distances[idx];
        //};
    };

    DistanceMatrix.prototype.print = function() {
        console.log(this);
    };

    return DistanceMatrix;
});
