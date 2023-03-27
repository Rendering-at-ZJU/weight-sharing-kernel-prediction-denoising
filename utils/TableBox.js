var TableBox = function(parent, stats) {

    var h1 = document.createElement("h1");
    h1.className = "title";
    h1.appendChild(document.createTextNode("Error Metrics"));
    parent.appendChild(h1);

    for (var i = 0; i < stats.length; ++i) {

        var table = document.createElement("table"); 
        table.className = "stats";

        var caption = document.createElement("caption");
        caption.className = "stats";
        caption.appendChild(document.createTextNode(stats[i]['title']));
        table.appendChild(caption);

        var thead = document.createElement("thead");
        thead.className = "stats";
        var tbody = document.createElement("tbody");
        tbody.className = "stats";

        table.appendChild(thead);
        table.appendChild(tbody);

        var tr = document.createElement("tr");
        tr.className = "stats";
        var td = document.createElement("th");
        td.className = "stats";
        td.appendChild(document.createTextNode('SPP'))
        tr.appendChild(td);
        for (var j = 0; j < stats[i]['labels'].length; ++j) {
            var td = document.createElement("th");
            td.className = "stats";
            td.appendChild(document.createTextNode(stats[i]['labels'][j]))
            tr.appendChild(td);
        }
        thead.appendChild(tr);

        for (var j = 0; j < stats[i]['series'].length; ++j) {
            var tr = document.createElement("tr");
            tr.className = "stats";
            var td = document.createElement("td");
            td.className = "stats";
            if (j == 0) {
                td.width = 100;
            }
            td.appendChild(document.createTextNode(stats[i]['series'][j]['label']))
            tr.appendChild(td);
            for (var k = 0; k < stats[i]['series'][j]['data'].length; ++k) {
                var td = document.createElement("td");
                td.className = "stats";
                td.appendChild(document.createTextNode(stats[i]['series'][j]['data'][k].toPrecision(4)))
                tr.appendChild(td);
            }
            tbody.appendChild(tr);
        }
        parent.appendChild(table);
    }
}
