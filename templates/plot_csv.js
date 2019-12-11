define(["Plotly"], function(Plotly) {
    return {
        greet: function(csv_filename) {
            function makeplot() {
                Plotly.d3.csv(csv_filename, function(data){ processData(data) });
            }

            function processData(allRows) {
                // Keys:
                // iteration, training loss, test loss, train fscore, test fscore
                // train fscore and test fscore are on a different scale.

                row = allRows[0];
                keys = Object.keys(row);

                // construct a dictionary of arrays of the keys.
                data = {};
                for(var i in keys) {
                    data[keys[i]] = [];
                }
                console.log(data);

                // load the data into the arrays.
                for(var i = 0; i < allRows.length; i++) {
                    row = allRows[i];

                    // loop over the keys and add the data.
                    for(var j in keys) {
                        data[keys[j]].push(+row[keys[j]]);
                    }
                }

                // console.log(data["training loss"])
                makeplotly(data);
            }

            function makeplotly(data) {
                var plotDiv = document.getElementById("plot");
                keys = Object.keys(data);
                traces = [];
                for(var i = 1; i < keys.length; i++) {
                    traces.push(
                        {
                            x: data[keys[0]],
                            y: data[keys[i]],
                            name: keys[i]
                        }
                    )
                }
                console.log(traces);
                var layout = {
                    xaxis: {
                        autorange: true
                    },
                    yaxis: {
                        title: "Loss",
                        //type: "log",
                        autorange: true
                    }
                };
                Plotly.newPlot("myDiv", traces, layout)
            }

            makeplot();
        }
    }
});