define(["d3", "d3-color", "d3-format", "d3-interpolate", "d3-scale-chromatic"], function(d3, d3_format) {
    return {
        greet: function(csv_filename, movie_name, fps) {
            var margin = {
                top: 20,
                right: 20,
                bottom: 110,
                left: 50
            };
            var margin2 = {
                    top: 430,
                    right: 20,
                    bottom: 30,
                    left: 40
                };
            var width = 960 - margin.left - margin.right;
            var height = 500 - margin.top - margin.bottom;
            var height2 = 500 - margin2.top - margin2.bottom;

            var x = d3.scaleLinear().range([0, width]);
            var y = d3.scaleLinear().range([height, 0]);
            var x2 = d3.scaleLinear().range([0, width]);
            var y2 = d3.scaleLinear().range([height2, 0]);

            var xAxis = d3.axisBottom(x);
            var yAxis = d3.axisLeft(y);
            var xAxis2 = d3.axisBottom(x2);

            // global path variables
            var paths_focus = [];
            var paths_context = [];

            // create an svg to plot within.
            var svg = d3.select("body")
                .append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)

            svg.append("defs").append("clipPath")
                .attr("id", "clip")
                .append("rect")
                .attr("width", width)
                .attr("height", height);

            var focus = svg.append("g")
                .attr("class", "focus")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

            var context = svg.append("g")
                .attr("class", "context")
                .attr("transform", "translate(" + margin2.left + "," + margin2.top + ")");

            var brush = d3.brushX()
                .extent([
                    [0, 0],
                    [width, height2]
                ])
                .on("brush", brushed);

            var zoom = d3.zoom()
                .scaleExtent([1, Infinity])
                .translateExtent([
                    [0, 0],
                    [width, height]
                ])
                .extent([
                    [0, 0],
                    [width, height]
                ])
                .on("zoom", zoomed);

            var frame_line = [];
            var mouse_line = [];

            d3.csv(csv_filename, function(error, data) {
                // console.log("hi");
                if(error) throw error;
                var headerNames = d3.keys(data[0]);
                // hack, remove the image header from the list of headers
                idx = headerNames.indexOf("image");
                if(idx != undefined) {
                    // console.log("hi!");
                    headerNames.splice(idx, 1);
                }
                // console.log(headerNames);

                var scores = new Array();
                data.forEach(function(d) {
                    // scores.push(d);
                    d.frame = +d.frame;
                    // store the values in an array
                    d.values = [];
                    for(let i = 0; i < headerNames.length; i++) {
                        d.values.push(+d[headerNames[i]]);
                    }
                });
                // console.log(data);

                var divs = [];
                for(var i = 1; i < headerNames.length; i++) {
                    divs.push(d3.select("body").append("div")
                        .attr("class", "tooltip")
                        .style("opacity", 0));
                }

                // setup the extents.
                data.sort(function(a, b) {
                    return d3.ascending(a.frame, b.frames);
                });
                x.domain(d3.extent(data, function(d) {
                    return d.frame;
                }));
                y.domain([-0.1, 1.1]);
                x2.domain(x.domain());
                y2.domain(y.domain());

                var color = d3.scaleSequential(d3.interpolateRainbow);
                paths_focus = create_paths(headerNames, focus, x, y, color, data);

                // Add the X Axis
                // focus.append("g")
                //     .attr("transform", "translate(0," + height + ")")
                //     .call(d3.axisBottom(x));
                focus.append("g")
                    .attr("class", "axis axis--x")
                    .attr("transform", "translate(0," + height + ")")
                    .call(xAxis);

                // Add the Y Axis
                focus.append("g")
                    .call(d3.axisLeft(y));

                focus.append("rect")
                    .attr('class', 'overlay')
                    .attr('width', width)
                    .attr('height', height)
                //    .on('mouseover', mouseover)
                //    .on('mouseout', mouseout) // () => drop_line.style('display', 'none'))
                //    .on('mousemove', mousemove);

                // create the zoom div
                context.append("g")
                    .attr("class", "axis axis--x")
                    .attr("transform", "translate(0," + height2 + ")")
                    .call(xAxis2);

                context.append("g")
                    .attr("class", "brush")
                    .call(brush)
                    .call(brush.move, x.range());

                // create the zoom viewport?
                svg.append("rect")
                    .attr("class", "zoom")
                    .attr("width", width)
                    .attr("height", height)
                    .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
                    .call(zoom)
                    .on('mouseover', mouseover)
                    .on('mouseout', mouseout) // () => drop_line.style('display', 'none'))
                    .on('mousemove', mousemove);

                paths_context = create_paths(headerNames, context, x2, y2, color, data);

                // create the video div
                var video_div = document.createElement("video");
                if (video_div.canPlayType("video/mp4")) {
                    // video_div.setAttribute("src", "moocow.mp4");
                    // video_div.setAttribute("src", "Knot_Tying_B001_capture1.avi");
                    video_div.setAttribute("src", movie_name);
                    //video_div.setAttribute("type", "video/mp4");
                }

                video_div.setAttribute("width", "704");
                video_div.setAttribute("height", "260");
                video_div.setAttribute("controls", "controls");

                video_div.setAttribute("x", 125)
                video_div.setAttribute("y", 430);
                //video_div.setAttribute("class", "video-js");
                document.body.appendChild(video_div);

                video_div.currentTime = 1.0 * 100 / fps;

                // this is the vertical line
                mouse_line = focus.append("path")
                    .attr("class", "mouse_line")
                    .style("stroke", "blue")
                    .style("stroke-width", "1px")
                    .style("opacity", "0.5");
                // move the vertical line
                //d3.select(".mouse_line")
                mouse_line.attr("d", function() {
                    //var d = "M" + mouse[0] + "," + height;
                    //d += " " + mouse[0] + "," + 0;
                    d = "M " + x(100) + " " + height;
                    d += " " + x(100) + " 0";
                    return d;
                });
                // console.log(d3.event.pageX);
                console.log(focus)

                frame_line = focus.append("path")
                    .attr("class", "frame_line")
                    .style("stroke", "gray")
                    .style("stroke-width", "1px")
                    .style("opacity", "0.5");


                video_div.playbackRate = 1;
                console.log(video_div.currentTime);

                video_div.onplay = function() {
                    setInterval(function() {
                        // mouse_pos = d3.mouse(this);
                        frame = video_div.currentTime * fps;
                        frame_line.attr("d", function() {
                            //var d = "M" + mouse[0] + "," + height;
                            //d += " " + mouse[0] + "," + 0;
                            d = "M " + x(frame) + " " + height;
                            d += " " + x(frame) + " 0";
                            return d;
                        });

                        var bisector = d3.bisector(d => d.values[0]).left;
                        const frame_idx = bisector(data, frame, 1);
                        for(var i = 1; i < headerNames.length; i++) {
                            //console.log(d3.event.pageX)
                            offset = 30 + (i - 1) * 40;
                            divs[i - 1].transition()
                                .duration(200)
                                .style("opacity", .9);
                            divs[i - 1].html(data[frame_idx].values[0] + "<br/>" + data[frame_idx].values[i])
                                .style("left", (x(frame) + 80) + "px")
                                .style("top", (y(1.0) - 28 + offset) + "px")
                                //.style("left", (x(d.frame) + 60) + "px")
                                //.style("top", (y(d.value) - 15 - offset) + "px")
                                .style("background", color((i - 1) / (headerNames.length - 1)));
                        }
                    }, 1/fps);
                };


                // setup the legend
                create_legend(focus, color, headerNames);

                function mouseover() {
                }

                function mouseout() {
                    // for(var i = 1; i < headerNames.length; i++) {
                    //     divs[i - 1].transition()
                    //         .duration(200)
                    //         .style("opacity", 0);
                    // }
                }

                function mousemove() {
                    const x0 = x.invert(d3.mouse(this)[0]);
                    // const y0 = d3.mouse(this)[1];
                    // console.log(x0);
                    var bisector = d3.bisector(d => d.values[0]).left;
                    const frame_idx = bisector(data, x0, 1);
                    video_div.currentTime = frame_idx / fps;
                    // console.log(data[frame_idx].values[1]);
                    // show the divs
                    for(var i = 1; i < headerNames.length; i++) {
                        offset = 30 + (i - 1) * 40;
                        divs[i - 1].transition()
                        .duration(200)
                        .style("opacity", .9);
                        divs[i - 1].html(data[frame_idx].values[0] + "<br/>" + data[frame_idx].values[i])
                        .style("left", (x(x0) + 80) + "px")
                        .style("top", (y(1.0) - 28 + offset) + "px")
                        //.style("left", (x(d.frame) + 60) + "px")
                        //.style("top", (y(d.value) - 15 - offset) + "px")
                        .style("background", color((i - 1) / (headerNames.length - 1)));
                    }

                    mouse_line.attr("d", function() {
                        d = "M " + x(x0) + " " + height;
                        d += " " + x(x0) + " 0";
                        return d;
                    });
                    frame_line.attr("d", function() {
                        //var d = "M" + mouse[0] + "," + height;
                        //d += " " + mouse[0] + "," + 0;
                        d = "M " + x(x0) + " " + height;
                        d += " " + x(x0) + " 0";
                        return d;
                    });
                }

            }); // end d3.data

            function brushed() {
                if (d3.event.sourceEvent && d3.event.sourceEvent.type === "zoom") return; // ignore brush-by-zoom
                var selection = d3.event.selection || x2.range();
                //var s = d3.event.selection || x2.range();
                x.domain(selection.map(x2.invert, x2));

                for(var i = 0; i < paths_focus.length; i++) {
                    //console.log(paths_main[i]);
                    var line_generator = d3.line()
                        .curve(d3.curveLinear)
                        .x(function(d) {
                            return x(d.values[0]);
                        })
                        .y(function(d) {
                            var idx = i + 1;
                            return y(d.values[idx]);
                        });
                    paths_focus[i].attr("d", line_generator);
                    // paths_main[i].attr("d", function(d) { return line_generator(d.values); });
                }

                focus.select(".axis--x").call(xAxis);
                svg.select(".zoom").call(zoom.transform, d3.zoomIdentity
                    .scale(width / (selection[1] - selection[0]))
                    .translate(-selection[0], 0));
            }

            function zoomed() {
                if (d3.event.sourceEvent && d3.event.sourceEvent.type === "brush") return; // ignore zoom-by-brush
                var t = d3.event.transform;
                x.domain(t.rescaleX(x2).domain());

                for(var i = 0; i < paths_focus.length; i++) {
                    //console.log(paths_main[i]);
                    var line_generator = d3.line()
                        .curve(d3.curveLinear)
                        .x(function(d) {
                            return x(d.values[0]);
                        })
                        .y(function(d) {
                            var idx = i + 1;
                            return y(d.values[idx]);
                        });
                    paths_focus[i].attr("d", line_generator);
                    // paths_main[i].attr("d", function(d) { return line_generator(d.values); });
                }

                focus.select(".axis--x").call(xAxis);
                context.select(".brush").call(brush.move, x.range().map(t.invertX, t));
            }

            function create_paths(headerNames, svg, x, y, color, data) {
                var paths = [];
                // Add the valueline path.
                // Start at "i = 1", i = 0 is the x axis.
                for(let i = 1; i < headerNames.length; i++) {
                    // construct the line function
                    //var yellow = d3.scaleSequential(d3.interpolatePiYG);
                    //console.log(yellow);
                    var line_generator = d3.line()
                        .curve(d3.curveLinear)
                        .x(function(d) {
                            return x(d.values[0]);
                        })
                        .y(function(d) {
                            var idx = i;
                            return y(d.values[idx]);
                        });
                    paths.push(svg.append("path")
                        .data([data])
                        .attr("clip-path", "url(#clip)")
                        .attr("class", "line")
                        .style("stroke", function() { // Add dynamically
                            return color((i - 1) / (headerNames.length - 1)); })
                            .attr("d", line_generator))
                }
                return paths;
            }

            function create_legend(focus, color, headerNames) {
                var legendRectSize = 18;
                var legendSpacing = 4;

                var data_names = headerNames.slice()
                data_names.shift();
                var legend = focus.selectAll('.legend')
                .data(data_names)
                .enter()
                .append('g')
                .attr('class', 'legend')
                .attr('transform', function (d, i) {
                    // console.log(d);
                    var height = legendRectSize + legendSpacing;
                    var offset = height * data_names.length / 2;
                    var horz = legendRectSize;
                    var vert = i * height;
                    return 'translate(' + horz + ',' + vert + ')';
                });

                legend.append('rect')
                .attr('width', legendRectSize)
                .attr('height', legendRectSize)
                .style('fill', function (d) {
                    return color(data_names.indexOf(d) / data_names.length)
                })
                .style('stroke', function (d) {
                    return color(data_names.indexOf(d) / data_names.length)
                });
                legend.append('text')
                .attr('x', legendRectSize + legendSpacing)
                .attr('y', legendRectSize - legendSpacing)
                .text(function (d) {
                    return d;
                });
            }
        }
    }
})
