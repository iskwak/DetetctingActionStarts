// contents of main.js:
require.config({
    paths: {
        Plotly: "https://cdn.plot.ly/plotly-latest.min"
    }
});

requirejs(["Plotly"],
    function(Plotly) {
        function makeplot() {
            Plotly.d3.csv("loss.csv", function(data){ processData(data) });
        }

        function processData(allRows) {
            console.log(allRows);

            // Keys:
            // iteration, training loss, test loss, train fscore, test fscore
            // train fscore and test fscore are on a different scale.
            var x = [], train_loss = [], test_loss = [],
                train_fscore = [], test_fscore = [],
                train_lift = [], test_lift = [],
                train_hand = [], test_hand = [],
                train_grab = [], test_grab = [],
                train_sup = [], test_sup = [],
                train_mouth = [], test_mouth = [],
                train_chew = [], test_chew = [];

            // load the data into the arrarys
            for(var i = 0; i < allRows.length; i++) {
                row = allRows[i];
                console.log(row)
                x.push(row["iteration"]);
                train_loss.push(row["training loss"]);
                test_loss.push(row["test loss"]);
                train_fscore.push(row["train fscore"]);
                test_fscore.push(row["test fscore"]);

                train_lift.push(row["train lift"]);
                train_hand.push(row["train hand"]);
                train_grab.push(row["train grab"]);
                train_sup.push(row["train sup"]);
                train_mouth.push(row["train mouth"]);
                train_chew.push(row["train chew"]);

                test_lift.push(row["test lift"]);
                test_hand.push(row["test hand"]);
                test_grab.push(row["test grab"]);
                test_sup.push(row["test sup"]);
                test_mouth.push(row["test mouth"]);
                test_chew.push(row["test chew"]);
            }
            console.log(train_loss)
            makeplotly(x, train_loss, test_loss, train_fscore, test_fscore,
                       train_lift, train_hand, train_grab, train_sup,
                       train_mouth, train_chew,
                       test_lift, test_hand, test_grab, test_sup,
                       test_mouth, test_chew);
        }

        function makeplotly(x, train_loss, test_loss,
                            train_fscore, test_fscore,
                            train_lift, train_hand, train_grab, train_sup,
                            train_mouth, train_chew,
                            test_lift, test_hand, test_grab, test_sup,
                            test_mouth, test_chew) {
            var plotDiv = document.getElementById("plot");
            var traces = 
                [{
                    x: x,
                    y: train_loss,
                    name: "Training Loss",
                },
                {
                    x: x,
                    y: test_loss,
                    name: "Testing Loss",
                },
                {
                    x: x,
                    y: train_fscore,
                    name: "Mean Training F-score",
                    yaxis: "y2"
                },
                {
                    x: x,
                    y: test_fscore,
                    name: "Mean Test F-score",
                    yaxis: "y2"
                },
                {
                    x: x,
                    y: train_lift,
                    name: "Training Lift F-score",
                    yaxis: "y2"
                },
                {
                    x: x,
                    y: train_hand,
                    name: "Training Hand F-score",
                    yaxis: "y2"
                },
                {
                    x: x,
                    y: train_grab,
                    name: "Training Grab F-score",
                    yaxis: "y2"
                },
                {
                    x: x,
                    y: train_sup,
                    name: "Training Suppinate F-score",
                    yaxis: "y2"
                },
                {
                    x: x,
                    y: train_mouth,
                    name: "Training Mouth F-score",
                    yaxis: "y2"
                },
                {
                    x: x,
                    y: train_chew,
                    name: "Training Chew F-score",
                    yaxis: "y2"
                },
                {
                    x: x,
                    y: test_lift,
                    name: "Test Lift F-score",
                    yaxis: "y2"
                },
                {
                    x: x,
                    y: test_hand,
                    name: "Test Hand F-score",
                    yaxis: "y2"
                },
                {
                    x: x,
                    y: test_grab,
                    name: "Test Grab F-score",
                    yaxis: "y2"
                },
                {
                    x: x,
                    y: test_sup,
                    name: "Test Suppinate F-score",
                    yaxis: "y2"
                },
                {
                    x: x,
                    y: test_mouth,
                    name: "Test Mouth F-score",
                    yaxis: "y2"
                },
                {
                    x: x,
                    y: test_chew,
                    name: "Test Chew F-score",
                    yaxis: "y2"
                },
                ]

            var layout = {
                xaxis: {
                    autorange: true
                },
                yaxis: {
                    title: "Loss",
                    type: "log",
                    autorange: true
                },
                yaxis2: {
                    title: "FP",
                    overlaying: "y",
                    side: "right",
                    range: [0, 1]
                }
            };
            Plotly.newPlot("myDiv", traces, layout)
        }

        makeplot();
    });
