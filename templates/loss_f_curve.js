// contents of main.js:
require.config({
    paths: {
        Plotly: "https://cdn.plot.ly/plotly-latest.min"
    }
});

requirejs(["Plotly"],
    function(Plotly) {
        function makeplot() {
            Plotly.d3.csv("loss_f.csv", function(data){ processData(data) });
        }

        function processData(allRows) {
            console.log(allRows);

            // Keys:
            // iteration, training loss, test loss, train fscore, test fscore
            // train fscore and test fscore are on a different scale.
            var train_loss = [], test_loss = [],
                train_fscore = [], test_fscore = [];

            // load the data into the arrarys
            for(var i = 0; i < allRows.length; i++) {
                row = allRows[i];
                console.log(row);
                train_loss.push(row["training loss"]);
                test_loss.push(row["test loss"]);
                train_fscore.push(row["train f1"]);
                test_fscore.push(row["test f1"]);
            }
            console.log(train_loss);
            console.log(train_fscore);
            makeplotly(train_loss, test_loss, train_fscore, test_fscore);
        }

        function makeplotly(train_loss, test_loss,
                            train_fscore, test_fscore) {
            var plotDiv = document.getElementById("plot");
            var traces = 
                [{
                    x: train_loss,
                    y: train_fscore,
                    name: "Training Loss",
                    mode: "markers",
                    type: "scatter",
                },
                {
                    x: test_loss,
                    y: test_fscore,
                    name: "Testing Loss",
                    mode: "markers",
                    type: "scatter",
                },
                ]

            var layout = {
                xaxis: {
                    autorange: true,
                    title: "Loss"
                },
                yaxis: {
                    title: "F Score",
                    // type: "log",
                    autorange: true
                },
            };
            Plotly.newPlot("myDiv", traces, layout)
        }

        makeplot();
    });
