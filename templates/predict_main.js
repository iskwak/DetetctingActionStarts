define(["d3", "d3-format"], function(d3, d3_format) {
    return {
        greet: function(csv_filename) {
            var margin = {
                    top: 20,
                    right: 20,
                    bottom: 110,
                    left: 50
                },
                margin2 = {
                    top: 430,
                    right: 20,
                    bottom: 30,
                    left: 40
                },
                width = 960 - margin.left - margin.right,
                height = 500 - margin.top - margin.bottom,
                height2 = 500 - margin2.top - margin2.bottom;

            var parseDate = d3.timeParse("%b %Y");

            var x = d3.scaleLinear().range([0, width]),
                x2 = d3.scaleLinear().range([0, width]),
                y = d3.scaleLinear().range([height, 0]),
                y2 = d3.scaleLinear().range([height2, 0]),
                z = d3.scaleOrdinal(d3.schemeCategory10);

            const bisectDate = d3.bisector(d => d.frame).left;

            var xAxis = d3.axisBottom(x),
                xAxis2 = d3.axisBottom(x2),
                yAxis = d3.axisLeft(y);

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

            var svg = d3.select("body").append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom);


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

            var line_main = d3.line()
                .curve(d3.curveLinear)
                .x(function(d) {
                    return x(d.frame);
                })
                .y(function(d) {
                    return y(d.value);
                });
            var paths_main = [];
            var line_context = d3.line()
                .curve(d3.curveLinear)
                .x(function(d) {
                    return x2(d.frame);
                })
                .y(function(d) {
                    return y2(d.value);
                });

            d3.csv(csv_filename, type, function(error, data) {
                if (error) throw error;
                var headerNames = d3.keys(data[0]);
                console.log(headerNames);
                // console.log(data);
                data.sort(function(a, b) {
                    return d3.ascending(a.frame, b.frame);
                });

                x.domain(d3.extent(data, function(d) {
                    return d.frame;
                }));
                y.domain([-.1, 1.1]);
                x2.domain(x.domain());
                y2.domain(y.domain());

                var scores = data.columns.slice(1).map(function(id) {
                    return {
                        id: id,
                        values: data.map(function(d) {
                            return {
                                frame: d.frame,
                                value: d[id],
                                image: d.image
                            };
                        })
                    };
                });

                // append scatter plot to main chart area 
                //console.log(scores[0].values);
                var divs = [];
                for(var i = 0; i < scores.length; i++) {
                    divs.push(d3.select("body").append("div")
                        .attr("class", "tooltip")
                        .style("opacity", 0));
                }
                var img_div = d3.select("body").append("div");

                for(var i = 0; i < scores.length; i++) {
                    if(scores[i].id != "image") {
                        paths_main.push(add_draw_paths(scores, i, focus, line_main));
                    }
                }

                var dots_main = [];
                for(var i = 0; i < scores.length; i++) {
                    if(scores[i].id != "image") {
                        dots_main.push(add_draw_points(scores, i, focus, x, y));
                    }
                }

                focus.append("g")
                    .attr("class", "axis axis--x")
                    .attr("transform", "translate(0," + height + ")")
                    .call(xAxis);

                focus.append("g")
                    .attr("class", "axis axis--y")
                    .call(yAxis);

                focus.append("text")
                    .attr("transform", "rotate(-90)")
                    .attr("y", 0 - margin.left)
                    .attr("x", 0 - (height / 2))
                    .attr("dy", "1em")
                    .style("text-anchor", "middle")
                    .text("Score");

                svg.append("text")
                    .attr("transform",
                        "translate(" + ((width + margin.right + margin.left) / 2) + " ," +
                        (height + margin.top + margin.bottom) + ")")
                    .style("text-anchor", "middle")
                    .text("Frame");

                // append scatter plot to brush chart area
                // the points are kind of messy in the context     
                // var dots_context = [];
                // for(var i = 0; i < scores.length; i++) {
                //     dots_context.push(add_draw_points(scores, i, context, x2, y2))
                // }
                var paths_context = [];
                for(var i = 0; i < scores.length; i++) {
                    paths_context.push(add_draw_paths(scores, i, context, line_context))
                }

                context.append("g")
                    .attr("class", "axis axis--x")
                    .attr("transform", "translate(0," + height2 + ")")
                    .call(xAxis2);

                context.append("g")
                    .attr("class", "brush")
                    .call(brush)
                    .call(brush.move, x.range());

                var drop_lines = [];
                for(var i = 0; i < scores.length; i++) {
                    drop_lines.push(focus.append('g')
                        .attr('class', 'drop_line')
                        .style('display', 'none'));

                    drop_lines[i].append('circle')
                        .attr('r', 4)
                        .attr("fill", "none")
                        .attr("stroke", "black");

                    drop_lines[i].append('line')
                        .classed('y', true)
                        .attr('fill', 'none')
                        .attr('stroke', 'black')
                        .attr('stroke-width', '1.5px')
                        .attr('stroke-dasharray', '3 3');

                    drop_lines[i].append('line')
                        .classed('y', true);
                }
                d3.selectAll('.drop_line')
                    .style('opacity', 0.7);

                // drop_line.append('text')
                //     .attr('x', 9)
                //     .attr('dy', '-.35em');

                svg.append("rect")
                    .attr("class", "zoom")
                    .attr("width", width)
                    .attr("height", height)
                    .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
                    .call(zoom)
                    .on('mouseover', mouseover)
                    .on('mouseout', mouseout) // () => drop_line.style('display', 'none'))
                    .on('mousemove', mousemove);

                function mouseover() {
                    for(var i = 0; i < scores.length; i++)
                        drop_lines[i].style('display', null)
                }

                function mouseout() {
                    for(var i = 0; i < scores.length; i++)
                        drop_lines[i].style('display', 'none');
                    for(var i = 0; i < scores.length; i++)
                        divs[i].transition()
                            .duration(200)
                            .style("opacity", 0);
                }

                function mousemove() {
                    const x0 = x.invert(d3.mouse(this)[0]);
                    create_drop_line(scores, 0, x0, 0);
                    create_drop_line(scores, 1, x0, 50);
                }

                function create_drop_line(scores, idx, x0, offset) {
                    const i = bisectDate(scores[idx].values, x0, 1);

                    const d0 = scores[idx].values[i - 1];
                    const d1 = scores[idx].values[i];
                    const d = x0 - d0.frame > d1.frame - x0 ? d1 : d0;
                    drop_lines[idx].attr('transform', `translate(${x(d.frame)}, ${y(d.value)})`);
                    drop_lines[idx].select('line.x')
                        .attr('x1', 0)
                        .attr('x2', -x(d.frame))
                        .attr('y1', 0)
                        .attr('y2', 0);

                    drop_lines[idx].select('line.y')
                        .attr('x1', 0)
                        .attr('x2', 0)
                        .attr('y1', 0)
                        .attr('y2', height - y(d.value));

                    //drop_line.select('text').text(d3.format(".3f")(d.lift));
                    divs[idx].transition()
                        .duration(200)
                        .style("opacity", .9);
                    divs[idx].html(d.frame + "<br/>" + d.value)
                        .style("left", (d3.event.pageX + 10) + "px")
                        .style("top", (d3.event.pageY - 28 - offset) + "px")
                        //.style("left", (x(d.frame) + 60) + "px")
                        //.style("top", (y(d.value) - 15 - offset) + "px")
                        .style("background", z(scores[idx].id));

                    img_div.transition()        
                        //.duration(200);
                    //console.log(scores[idx].values[i]);
                    var string = "<img src=" + scores[idx].values[i].image + " />";
                    img_div.html(string) //this will add the image on mouseover
                        .style("left", (d3.event.pageX + 10) + "px")     
                        .style("top", (d3.event.pageY + 50) + "px")
                        .style("font-color", "white");
                }

                function add_draw_paths(scores, idx, space, line_func) {
                    var line = space.append("path").datum(scores[idx]);
                    line.attr("clip-path", "url(#clip)")
                        .attr("class", "line")
                        .attr("d", function(d) { return line_func(d.values); })
                        .style("stroke", function(d) { return z(d.id); });
                    //console.log(line);
                    return line;
                }

                function add_draw_points(scores, idx, space, d_x, d_y) {
                    //console.log(scores[idx])
                    var id = scores[idx].id;
                    //console.log(id)
                    // console.log(data)
                    var dots = space.append("g");
                        dots.attr("clip-path", "url(#clip)");
                        dots.selectAll("dot")
                            .data(scores[idx].values)
                            .enter().append("circle")
                            .attr('class', 'dot')
                            .attr("r", 1.5)
                            .style("opacity", .5)
                            .attr("cx", function(d) {
                                return d_x(d.frame);
                            })
                            .attr("cy", function(d) {
                                return d_y(d.value);
                            })
                            .style("stroke", () => z(id))
                            .style("fill", () => z(id));
                    return dots;
                }




            });

            //create brush function redraw scatterplot with selection
            function brushed() {
                if (d3.event.sourceEvent && d3.event.sourceEvent.type === "zoom") return; // ignore brush-by-zoom
                var selection = d3.event.selection || x2.range();
                //var s = d3.event.selection || x2.range();
                x.domain(selection.map(x2.invert, x2));

                for(var i = 0; i < paths_main.length; i++) {
                    //console.log(paths_main[i]);
                    paths_main[i].attr("d", function(d) { return line_main(d.values); });
                }

                focus.selectAll(".dot")
                    .attr("cx", function(d) {
                        return x(d.frame);
                    })
                    .attr("cy", function(d) {
                        return y(d.value);
                    });
                focus.select(".axis--x").call(xAxis);

                svg.select(".zoom").call(zoom.transform, d3.zoomIdentity
                    .scale(width / (selection[1] - selection[0]))
                    .translate(-selection[0], 0));
            }

            function zoomed() {
                if (d3.event.sourceEvent && d3.event.sourceEvent.type === "brush") return; // ignore zoom-by-brush
                var t = d3.event.transform;
                x.domain(t.rescaleX(x2).domain());

                for(var i = 0; i < paths_main.length; i++) {
                    //console.log(paths_main[i]);
                    paths_main[i].attr("d", function(d) { return line_main(d.values); });
                }

                focus.selectAll(".dot")
                    .attr("cx", function(d) {
                        return x(d.frame);
                    })
                    .attr("cy", function(d) {
                        return y(d.value);
                    });

                focus.select(".axis--x").call(xAxis);
                context.select(".brush").call(brush.move, x.range().map(t.invertX, t));
            }

            function type(d, _, columns) {
                // convert the strings into numerical values.
                for (var i = 0; i < columns.length; ++i) {
                    col_name = columns[i];
                    if(col_name == "image")
                        d[col_name] = d[col_name]
                    else
                        d[col_name] = +d[col_name];
                    // d[i-1] = +d[col_name];
                }
                return d;
            }
    	}
    }
});
