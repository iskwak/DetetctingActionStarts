// contents of main.js:
require.config({
    paths: {
        Plotly: "https://cdn.plot.ly/plotly-latest.min"
    }
});

requirejs(["Plotly"],
    function(Plotly) {
        function makeplot() {
            Plotly.d3.csv("temp.csv", function(data){ processData(data) });
        }

        function processData(allRows) {
            console.log(allRows);

            // Keys:
            // iteration, training loss, test loss, train fscore, test fscore
            // train fscore and test fscore are on a different scale.
            var threshold = [], lift = [], hand = [], grab = [],
                sup = [], mouth = [], chew = [];

            // load the data into the arrarys
            for(var i = 0; i < allRows.length; i++) {
                row = allRows[i];
                console.log(row)
                threshold.push(row["threshold"]);
                lift.push(row["lift"]);
                hand.push(row["hand"]);
                grab.push(row["grab"]);
                sup.push(row["supinate"]);
                mouth.push(row["mouth"]);
                chew.push(row["chew"]);
            }
            console.log(threshold)
            makeplotly(threshold, lift, hand, grab, sup, mouth, chew);
        }

        function makeplotly(threshold, lift, hand, grab, sup, mouth, chew) {
            var plotDiv = document.getElementById("plot");
            var traces = 
                [{
                    x: threshold,
                    y: lift,
                    name: "Lift",
                },
                {
                    x: threshold,
                    y: hand,
                    name: "Hand",
                },
                {
                    x: threshold,
                    y: grab,
                    name: "Grab",
                },
                {
                    x: threshold,
                    y: sup,
                    name: "Supinate",
                },
                {
                    x: threshold,
                    y: mouth,
                    name: "Mouth",
                },
                {
                    x: threshold,
                    y: chew,
                    name: "Chew"
                }];

            var layout = {
                xaxis: {
                    autorange: true
                },
                yaxis: {
                    title: "Mean F Score",
                    //type: "log",
                    autorange: true
                },
            };
            Plotly.newPlot("myDiv", traces, layout)
        }

        makeplot();
    });
