/**
 * @fileoverview The view for the triplet gui.
 * @author iskwak
 */
//define(["three","jquery","controls","datgui"], function(THREE,$,dat) {
define(["three","jquery","controls","datgui"], function(THREE,$) {

	var FOV = 45;
	var NEAR = 0.1;
	var FAR = 1000000;

	//var mouse;
	var ctrl = false;

    // helper functions to create visually pleasing colors
    var pastelColors = function pastelColors() {
        var r = (Math.round(random() * 127) + 127).toString(16);
        var g = (Math.round(random() * 127) + 127).toString(16);
        var b = (Math.round(random() * 127) + 127).toString(16);

        return '#' + r + g + b;
    };
    var seed = 1;
    function random() {
        var x = Math.sin(seed++) * 10000;
        return x - Math.floor(x);
    }

	var cluster_colors = [
        "FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF", "FFFFFF", 
        "800000", "008000", "000080", "808000", "800080", "008080", "808080"
        ];
	//for(var i = 0; i < 50; i++) {
	//	cluster_colors.push(pastelColors());
	//}


    var colors = [0x0000FF, 0x00FF00, 0xFF0000];
    var mode_params;

    _points = [];
    _orgspriteimgs = [];
    _reconspriteimgs = [];
    _showimgs = false;

	/** @constructor */
	var View = function(canvas,width,height,intersect) {
		/** @private */
		this._canvas = canvas;
		this._width = width;
		this._height = height;

		this._box = $("<div id=overlay></div>");
		this._box.css({
			position: "absolute",
			width: "200px",
			height: "200px",
			/*background: #44accf;*/
			background: "#a1a1a1",
			"z-index": "2",
			visibility: "hidden",
			border: "2px solid #a1a1a1",
			"border-radius": "25px",
		});
		this._canvas.append(this._box);

		this._intersect = null;

		this.isInitialized = false;
		this._mouse = {x:-1,y:-1};

		//this._points = [];
        //this._spriteimgs = [];

		/** @private */
		// three.js related variables
		this._renderer = {};
		this._camera = {};
		this._scene = {};
		this._controls = {};
		this._raycaster = {};
        this._orgdistmat = {};

        // dat gui
        this._gui = {};

        this._click = false;
        this._ctrl = false;

		/** @private */
		this._initRenderer = function _initRenderer() {
			this._renderer = new THREE.WebGLRenderer(
				{antialias: true});
			this._renderer.setSize(
				this._width,
				this._height);

			this._renderer.setPixelRatio(window.devicePixelRatio);

			//this._renderer.setClearColor(new THREE.Color(0xEEEEEE,0));
			this._renderer.setClearColor(new THREE.Color(0x000000,0));
			//this._renderer.autoClear = false;
			//this._renderer.clear();
		};

		/** @private */
		this._buildAxes = function _buildAxes() {
			var axes = new THREE.Object3D();

			var geometry = new THREE.Geometry();
			geometry.vertices.push(new THREE.Vector3(-500,0,0));
			geometry.vertices.push(new THREE.Vector3(500,0,0));
			var linesMaterial = new THREE.LineBasicMaterial({
				color: 0x787878,
				opacity: 0.2,
				linewidth: 0.05
			});
			var linesMaterial100 = new THREE.LineBasicMaterial({
				color: 0xFFFFFF,
				opacity: 0.2,
				linewidth: 0.05
			});
			var mainAxisMaterial = new THREE.LineBasicMaterial({
				color: 0xFF0000,
				opacity: 0.2,
				linewidth: 0.05
			});
			for(var i = 0; i <= 100; i++){
				offset = (i*10) - 500;
				var verline;
				var horline;
				if(offset === 0) {
					verline = new THREE.Line(geometry, mainAxisMaterial);
					horline = new THREE.Line(geometry, mainAxisMaterial);
				}
				else if(offset % 100 === 0) {
					verline = new THREE.Line(geometry, linesMaterial100);
					horline = new THREE.Line(geometry, linesMaterial100);
				}
                else {
					verline = new THREE.Line(geometry, linesMaterial);
					horline = new THREE.Line(geometry, linesMaterial);
                }

				horline.position.x = (i*10) - 500;
				horline.rotation.z = 90*Math.PI/180;

				axes.add(horline);

				verline.position.y = (i*10) - 500;
				verline.rotation.x = 90*Math.PI/180;

				axes.add(verline);
			}

			this._scene.add(axes);
		}; 


        /** @private */
        this._initializegui = function _initializegui() {
            var gui = new dat.GUI({
                resizable : false
            });


            function showClusterLabels() {
                for(var i = 0; i < _points.length; i++) {
                    _points[i].visible = true;
                    _orgspriteimgs[i].visible = false;
                    _reconspriteimgs[i].visible = false;
                }
            }
            function showOrgImages() {
                for(var i = 0; i < _points.length; i++) {
                    _points[i].visible = false;
                    _orgspriteimgs[i].visible = true;
                    _reconspriteimgs[i].visible = false;
                }
            }
            function showReconImages() {
                for(var i = 0; i < _points.length; i++) {
                    _points[i].visible = false;
                    _orgspriteimgs[i].visible = false;
                    _reconspriteimgs[i].visible = true;
                }
            }
            var params = function () {
                this["Mouse Over Select"] = true;
                this["Click Select"] = false;
                //this["Toggle Images"] = function() {toggleImages();};
                this["Show Cluster"] = true;
                this["Show Original"] = false;
                this["Show Recon"] = false;
            };

            mode_params = new params();
            var checkboxmouseover = gui.add(mode_params, 'Mouse Over Select', true).listen();
            var checkboxclick = gui.add(mode_params, 'Click Select', false).listen();

            var checkboxcluster = gui.add(mode_params, "Show Cluster", true).listen();
            var checkboxshoworg = gui.add(mode_params, "Show Original", false).listen();
            var checkboxshowrecon = gui.add(mode_params, "Show Recon", false).listen();

            checkboxmouseover.onChange(function(value) {
                if(mode_params["Click Select"] === true) {
                    mode_params["Click Select"] = false;
                }
                mode_params["Mouse Over Select"] = true;
            });

            checkboxclick.onChange(function(value) {
                if(mode_params["Mouse Over Select"] === true) {
                    mode_params["Mouse Over Select"] = false;
                }
                mode_params["Click Select"] = true;
            });

            checkboxcluster.onChange(function(value) {
                if(mode_params["Show Original"] === true) {
                    mode_params["Show Original"] = false;
                }
                if(mode_params["Show Recon"] === true) {
                    console.log(mode_params["Show Recon"]);
                    mode_params["Show Recon"] = false;
                    console.log(mode_params["Show Recon"]);
                }
                mode_params["Show Cluster"] = true;
                showClusterLabels();
            });
            checkboxshoworg.onChange(function(value) {
                if(mode_params["Show Cluster"] === true) {
                    mode_params["Show Cluster"] = false;
                }
                if(mode_params["Show Recon"] === true) {
                    mode_params["Show Recon"] = false;
                }
                mode_params["Show Original"] = true;
                showOrgImages();
            });
            checkboxshowrecon.onChange(function(value) {
                if(mode_params["Show Cluster"] === true) {
                    mode_params["Show Cluster"] = false;
                }
                if(mode_params["Show Original"] === true) {
                    mode_params["Show Original"] = false;
                }
                mode_params["Show Recon"] = true;
                showReconImages();
            });
        };

		/** @private */
		this._initCamera = function _initCamera() {
			this._camera = new THREE.PerspectiveCamera(
				FOV, this._width/this._height, NEAR, FAR);

			this._camera.position.x = 0;
			this._camera.position.y = 0;
			this._camera.position.z = 70;
			//this._camera.position.z = 300;
		};
		this._initControls = function _initControls() {
			this._controls = new THREE.TrackballControls( 
				this._camera, 
				this._renderer.domElement );
			//controls.rotateSpeed = 1.0;
			this._controls.noRotate = true; // for now 2d 
			//this._controls.noRotate = false; // 3d 
			this._controls.zoomSpeed = 1.2;
			this._controls.panSpeed = 0.8;
			this._controls.noZoom = false;
			this._controls.noPan = false;
			this._controls.staticMoving = true;
			this._controls.dynamicDampingFactor = 0.3;
            this._controls.enabled = true;
		};

        /** @private */
        this._updateIntersections = function _updateIntersections(intersects) {
            if(mode_params["Click Select"] === true) {
                if(intersects.length <= 0 || this._click === false) {
                    return;
                }
            }
            if(intersects.length > 0) {
                console.log(intersects[0].object.cluster);
                if(this._intersect != intersects[0].object) {
                    var intersect_idx = intersects[0].object.idx;
                    //console.log(intersect_idx);
                    for(var i = 0; i < _points.length; i++) {
                        var curr_idx = _points[i].idx;

                        var new_color = this.heat_map_color(
                                this._orgdistmat.distances[intersect_idx][curr_idx]);
                        _points[i].material.color.setHSL(
                                new_color[0], new_color[1], new_color[2]);
                    }


                    intersects[0].object.material.color.setHex(
                            colors[0]);

                    this._intersect = intersects[0].object;
                }
            } else {
                for(var j = 0; j < _points.length; j++) {
                    var org_color = parseInt(
                                cluster_colors[_points[j].cluster].replace(/^#/, ''), 16);
                    _points[j].material.color.setHex(org_color);
                }
                this._intersect = null;
            }
        };


        this.heat_map_color = function heat_map_color(value) {
            //var threshold = 800;
            var threshold = 5000;
            var h = 0;
            //console.log(this.upper_thresh);

            if(this.upper_thresh < value)
                h = 0;
            else
                h = (this.upper_thresh - value)/this.upper_thresh;

            //console.log(h);
            h = h*2/3;
            return [h,1,0.5];
        };
	}; // END constructor


	/**
	 * Initialize the view object. Will setup the renderer, camera and
	 * and scene. No parameters.
	 * @public
	 */
	View.prototype.initialize = function animate() {
		// create the renderer
		this._initRenderer();
		this._canvas.append(this._renderer.domElement);

		// create the camera
		this._initCamera();

		// create scene
		this._scene = new THREE.Scene();

		// setup trackball controls
		this._initControls();

		// add the camera to the scene this._scene.add(this._camera);

		// build the axis
		this._buildAxes();

		this._raycaster = new THREE.Raycaster();

		// register callbacks
		//$(document).mousemove(onDocumentMouseMove);
	    //$(document).mouseup(onDocumentMouseUp);
	    //$(document).mousedown(onDocumentMouseDown);

        // init dat gui
        this._initializegui();

		this._isInitialized = true;
		return true;
	};

	/**
	 * Render will call the webgl render function and additionally
	 * will place the objects in place
	 * @public
	 */
	View.prototype.render = function render() {
		//if(ctrl === true) {
		//	this._controls.update();
		//}
		//console.log(this._mouse);
        if(this._ctrl === true || mode_params["Mouse Over Select"] === true)
        {
            this._controls.enabled = true;
            this._controls.update();
        }
        else {
            this._controls.enabled = false;
        }

		this._raycaster.setFromCamera( this._mouse, this._camera );
		intersects = this._raycaster.intersectObjects( _points );
        this._updateIntersections(intersects);

		this._renderer.render(this._scene, this._camera);
	};


	/**
	 * Animate will do the render loop.
	 * @public
	 */
	View.prototype.animate = function animate() {
		requestAnimationFrame( this.animate.bind(this) );
		this.render();
	};


	/**
	 * Create threejs spheres based on a list of (x,y) coordinates.
	 * @public
	 */
	View.prototype.addPoints = function addPoints(points,orgdistmat) {
		var geometry = new THREE.SphereGeometry(0.2, 32, 32);
        //var geometry = new THREE.Geometry();
		for(var i = 0; i < points.length; i++) {
            //console.log(points[i].id);
            //console.log(cluster_colors[points[i].cluster]);
			var material = new THREE.MeshBasicMaterial({
				color: cluster_colors[points[i].cluster]
			});
			var object = new THREE.Mesh(geometry, material);

			object.position.x = points[i].vector[0];
			object.position.y = points[i].vector[1];
			object.position.z = 0;
			object.cluster = points[i].cluster;
            object.idx = points[i].idx;
			object.img = points[i].img;

            //var vertex = new THREE.Vector3();
            //vertex.x = points[i].vector[0];
            //vertex.y = points[i].vector[1];
            //vertex.idx = points[i].idx;
            //vertex.cluster = points[i].idx;

			_points.push(object);
            //geometry.vertices.push(vertex);
			this._scene.add(_points[i]);

            var map = THREE.ImageUtils.loadTexture(points[i].media_src);
            map.minFilter = THREE.NearestFilter;
            var mat = new THREE.SpriteMaterial({map: map});
            var sprite = new THREE.Sprite(mat);
            sprite.position.set(
                points[i].vector[0],
                points[i].vector[1],
                0);

            //sprite.visible = true;
            sprite.visible = false;
            this._scene.add(sprite);
            _orgspriteimgs.push(sprite);


            var map2 = THREE.ImageUtils.loadTexture(points[i].media_src2);
            map2.minFilter = THREE.NearestFilter;
            var mat2 = new THREE.SpriteMaterial({map: map2});
            var sprite2 = new THREE.Sprite(mat2);
            sprite2.position.set(
                points[i].vector[0],
                points[i].vector[1],
                0);

            sprite2.visible = false;
            this._scene.add(sprite2);
            _reconspriteimgs.push(sprite2);

		}
        //this._scene.add(geometry);
        this._orgdistmat = orgdistmat;

        this.lower_thresh = orgdistmat.mean - 2*orgdistmat.std;
        if(this.lower_thresh < 0) {
            console.log('lower thresh is negative');
            this.lower_thresh = 0;
        }
        this.upper_thresh = orgdistmat.mean + 2*orgdistmat.std;
        //this.upper_thresh = orgdistmat.mean;// + 2*orgdistmat.std;
        console.log('upper thresh: ' + this.upper_thresh);
        console.log('lower thresh: ' + this.lower_thresh);
	};


	/**
	 * Callback for handling mouse movements.
	 * @public
	 */
	View.prototype.onMouseMove = function onMouseMove(event) {
        this._mouse = this.convertScreenToCoord(event.clientX, event.clientY);
		if(event.ctrlKey === true) {
			this._ctrl = true;
		} else {
			this._ctrl = false;
		}
    };


	View.prototype.onMouseDown = function onMouseDown(event) {
        if(event.button === 0){
            this._click = true;
        }
    };


	View.prototype.onMouseUp = function onMouseUp(event) {
		//if(event.ctrlKey === true) {
		//	ctrl = true;
		//} else {
		//	ctrl = false;
		//}
        this._click = false;
    };

	/**
	 * Callback for handling window resize events.
	 * @public
	 */
    View.prototype.onWindowResize = function onWindowResize(width, height) {
        //console.log(width + " " + height);
        this._width = width;
        this._height = height;
        this._camera.aspect = this._width/this._height;
        this._camera.updateProjectionMatrix();

        this._renderer.setSize(
            this._width,
            this._height);

        this.render();
    };


	/**
	 * Helper function for converting screen coordinates to point cloud
     * coordinates. (might be better as a private helper)
	 * @public
	 */
    View.prototype.convertScreenToCoord = function convertScreenToCoord(x,y) {
        var temp = {x:0, y:0};
        temp.x = (x/this._width) * 2 - 1;
        temp.y = -(y/this._height) * 2 + 1;

        return temp;
    };

	return View;
});




