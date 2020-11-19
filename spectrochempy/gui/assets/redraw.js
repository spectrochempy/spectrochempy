function redraw(){
    var figs = document.getElementsByClassName("js-plotly-plot")
    for (var i = 0; i < figs.length; i++) {
        Plotly.redraw(figs[i])
    }
}

setTimeout(function(){
    redraw();
}, 1000);