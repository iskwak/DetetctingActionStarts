/**
 * Given x and y data create a line graph
 */
function makePlotly(x,y,xlabel,ylabel) {
    var plotDiv = document.getElementById('plot');
    var traces = [{
        x: x,
        y: y,
        mode: 'lines+markers',
        type: 'scatter'
    }];

    var layout = {
        xaxis: {
            range: [0,5]
        },
        yaxis: {
            range: [5,25]
        },
        margin: {t: 20}
    };

    Plotly.newPlot('myDiv', traces, layout);

    //var img = canvas.toDataURL('image/png');
    var myDiv = document.getElementById('myDiv');
    //console.log(myDiv);
}

$(document).ready(function() {

    // The event listener for the file upload
    document.getElementById('txtFileUpload').addEventListener('change', upload, false);

    // Method that checks that the browser supports the HTML5 File API
    function browserSupportFileUpload() {

        var isCompatible = false;
        if (window.File && window.FileReader && window.FileList && window.Blob) {
            isCompatible = true;
        }
        return isCompatible;
    }

    // Method that reads and processes the selected file
    function upload(evt) {
        if (!browserSupportFileUpload()) {
            alert('The File APIs are not fully supported in this browser!');
        } else {

            var data = null;
            var file = evt.target.files[0];
            var reader = new FileReader();
            reader.readAsText(file);
            reader.onload = function(event) {
                var csvData = event.target.result;
                lines = csvData.split('\n');
                //console.log(lines);
                if(lines && lines.length > 0) {
                    // parse the header
                    var x = [];
                    var y = [];
                    header = lines[0].split(',');
                    line_length = header.length;
                    for(var i = 1; i < lines.length; i++) {
                        //console.log(i, ' ', lines[i]);
                        data = lines[i].split(',');
                        if(data.length == header.length) {
                            x.push(Number(data[0]));
                            y.push(Number(data[1]));
                        }
                    }
                    makePlotly(x,y,header[0],header[1]);
                }
                else {
                    alert('no data to import!');
                }
            };
            reader.onerror = function() {
                alert('Unable to read ' + file.fileName);
            };
        }
    }
});
