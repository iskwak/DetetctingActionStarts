require.config({
	baseUrl: "js",
	paths: {
		three: "libs/three.min",
		jquery: "libs/jquery-1.11.2",
		view: "view",
		point: "models/point",
        pointnd: "models/pointnd",
        distmat: "models/distancematrix",
		controls: "libs/TrackballControls",
        datgui: "libs/dat.gui.min"
	},
	shim: {
		three: {exports: 'THREE'},
		controls: {
			deps: ['three']
		}
		
	}
});

require(["jquery", "view","point","pointnd","distmat"], 
        function($, View, Point, PointND, DistanceMatrix) {
	//var mouse = {x: -1, y: -1};

    var scripts = document.getElementsByTagName('script');
    //var lastScript = scripts[scripts.length-1];
    var script = scripts[0];
    console.log(scripts[0]);
    console.log(script.getAttribute('data-type'));
    var type = script.getAttribute('data-type');

	// create a div to do everything in.
	glcanvas = $("<div id=glcanvas></div>");
	$("body").append(glcanvas);

    // create the view object and initialize it
	var view = new View(
		glcanvas,
		$(document).width(),
		$(document).height(),
		{});
	view.initialize();

    // register call backs 
    $(document).mousemove(function(event) {
        view.onMouseMove(event);
    });
    $(document).mousedown(function(event) {
        view.onMouseDown(event);
    });
    $(document).mouseup(function(event) {
        view.onMouseUp(event);
    });
    $(window).resize(function () {
        view.onWindowResize(
            window.innerWidth,
            window.innerHeight);
    });


    $.ajax({
        type: 'POST',
        url: 'getdata',
        data: 'beepboop',
        success: function(data) {
            //console.log('stuff');
            //console.log(data[0]);

            var proj_points = data[0];
            var org_points = data[1];

            var distmat = new DistanceMatrix(proj_points);
            var org_distmat = new DistanceMatrix(org_points);
            distmat.minus(org_distmat);

            view.addPoints(proj_points,org_distmat);
            //view.addPoints(proj_points,org_distmat);

            //view.addPoints(proj_points,org_distmat);

            view.animate();
        }
    });



});



	//$.ajax({
	//	type: 'POST',
	//	url: "getdata",
	//	data: "hello",
	//	success: function(dots){
	//		//console.log(msg);
	//		// setup the dots here.
	//		for(i = 0; i < dots.length; i++) {
	//			points.push(dots[i]);
	//		}
	//		view.addPoints(points);
	//		view.animate();
	//	}});
