/*!
    Wheelzoom 3.0.4
    license: MIT
    http://www.jacklmoore.com/wheelzoom
*/
var options = {
    zoom: 0.25,
    width: 1280
};

window.wheelzoom = (function(){

    var canvas = document.createElement('canvas');

    var main = function(img, options){
        if (!img || !img.nodeName || img.nodeName !== 'IMG') { return; }

        var width;
        var height;
        var previousEvent;
        var cachedDataUrl;

        function setSrcToBackground(img) {
            img.style.backgroundImage = 'url("'+img.src+'")';
            img.style.backgroundRepeat = 'no-repeat';
            canvas.width = options.width;
            canvas.height = img.naturalHeight;
            img.bgOffset = (canvas.width - img.naturalWidth)/2;
            cachedDataUrl = canvas.toDataURL();
            img.src = cachedDataUrl;
            img.style.backgroundSize = img.bgWidth+'px '+img.bgHeight+'px';
            img.style.backgroundPosition = img.bgOffset + img.bgPosX+'px '+img.bgPosY+'px';
        }

        function updateBgStyle() {
            if (img.bgPosX > 0) {
                img.bgPosX = 0;
            } else if (img.bgPosX < img.imWidth - img.bgWidth) {
                img.bgPosX = img.imWidth - img.bgWidth;
            }

            if (img.bgPosY > 0) {
                img.bgPosY = 0;
            } else if (img.bgPosY < height - img.bgHeight) {
                img.bgPosY = height - img.bgHeight;
            }

            img.style.backgroundSize = img.bgWidth+'px '+img.bgHeight+'px';
            img.style.backgroundPosition = img.bgOffset + img.bgPosX+'px '+img.bgPosY+'px';
        }

        function reset() {
            img.bgWidth = img.imWidth;
            img.bgHeight = img.imHeight;
            img.bgPosX = img.bgPosY = 0;
            updateBgStyle();
        }

        function onwheel(e) {
            var deltaY = 0;

            e.preventDefault();

            if (e.deltaY) { // FireFox 17+ (IE9+, Chrome 31+?)
                deltaY = -e.deltaY;
            } else if (e.wheelDelta) {
                deltaY = e.wheelDelta;
            }

            // As far as I know, there is no good cross-browser way to get the cursor position relative to the event target.
            // We have to calculate the target element's position relative to the document, and subtrack that from the
            // cursor's position relative to the document.
            var rect = img.getBoundingClientRect();
            var offsetX = e.pageX - rect.left - window.pageXOffset - img.bgOffset;
            var offsetY = e.pageY - rect.top - window.pageYOffset;

            // Record the offset between the bg edge and cursor:
            var bgCursorX = offsetX - img.bgPosX;
            var bgCursorY = offsetY - img.bgPosY;
            
            // Use the previous offset to get the percent offset between the bg edge and cursor:
            var bgRatioX = bgCursorX/img.bgWidth;
            var bgRatioY = bgCursorY/img.bgHeight;

            // Update the bg size:
            if (deltaY < 0) {
                img.bgWidth += img.bgWidth*options.zoom;
                img.bgHeight += img.bgHeight*options.zoom;
            } else {
                img.bgWidth -= img.bgWidth*options.zoom;
                img.bgHeight -= img.bgHeight*options.zoom;
            }

            // Take the percent offset and apply it to the new size:
            img.bgPosX = offsetX - (img.bgWidth * bgRatioX);
            img.bgPosY = offsetY - (img.bgHeight * bgRatioY);

            // Prevent zooming out beyond the starting size
            if (img.bgWidth <= img.imWidth || img.bgHeight <= img.imHeight) {
                reset();
            } else {
                updateBgStyle();
            }
        }

        function drag(e) {
            e.preventDefault();
            img.bgPosX += (e.pageX - previousEvent.pageX);
            img.bgPosY += (e.pageY - previousEvent.pageY);
            previousEvent = e;
            updateBgStyle();
        }

        function removeDrag() {
            document.removeEventListener('mouseup', removeDrag);
            document.removeEventListener('mousemove', drag);
        }

        // Make the background draggable
        function draggable(e) {
            e.preventDefault();
            previousEvent = e;
            document.addEventListener('mousemove', drag);
            document.addEventListener('mouseup', removeDrag);
        }

        function load() {
            if (img.src === cachedDataUrl) return;

            img.imWidth = img.naturalWidth;
            img.imHeight = img.naturalHeight;

            img.bgWidth = img.imWidth;
            img.bgHeight = img.imHeight;
            img.bgPosX = 0;
            img.bgPosY = 0;

            img.style.backgroundSize     = img.bgWidth+'px '+img.bgHeight+'px';
            img.style.backgroundPosition = img.bgPosX+' '+img.bgPosY;

            setSrcToBackground(img);

            img.addEventListener('wheelzoom.reset', reset);
            img.addEventListener('wheel', onwheel);
            img.addEventListener('mousedown', draggable);
        }

        var destroy = function (originalProperties) {
            img.removeEventListener('wheelzoom.destroy', destroy);
            img.removeEventListener('wheelzoom.reset', reset);
            img.removeEventListener('load', load);
            img.removeEventListener('mouseup', removeDrag);
            img.removeEventListener('mousemove', drag);
            img.removeEventListener('mousedown', draggable);
            img.removeEventListener('wheel', onwheel);

            img.style.backgroundImage = originalProperties.backgroundImage;
            img.style.backgroundRepeat = originalProperties.backgroundRepeat;
            img.src = originalProperties.src;
        }.bind(null, {
            backgroundImage: img.style.backgroundImage,
            backgroundRepeat: img.style.backgroundRepeat,
            src: img.src
        });

        img.addEventListener('wheelzoom.destroy', destroy);

        if (img.complete) {
            load();
        }

        img.addEventListener('load', load);
    };

    // Do nothing in IE8
    if (typeof window.getComputedStyle !== 'function') {
        return function(elements) {
            return elements;
        };
    } else {
        return function(elements, options) {
            if (elements && elements.length) {
                Array.prototype.forEach.call(elements, main, options);
            } else if (elements && elements.nodeName) {
                main(elements, options);
            }
            return elements;
        };
    }
}());



var ImageBox = function(parent, config) {
    var self = this;

    var box = document.createElement('div');
    box.className = "image-box";

    var h1 = document.createElement('h1');
    h1.className = "title";
    h1.appendChild(document.createTextNode("Images"));
    box.appendChild(h1);

    var help = document.createElement('div');
    help.appendChild(document.createTextNode("Use mouse wheel to zoom in/out, click and drag to pan."));
    help.className = "help";
    box.appendChild(help);

    this.tree = [];
    this.selection = [];
    this.buildTreeNode(config, 0, this.tree, box);

    for (var i = 0; i < this.selection.length; ++i) {
        this.selection[i] = 0;
    }
    this.showContent(0, 0);
    parent.appendChild(box);
    
    document.addEventListener("keypress", function(event) { self.keyPressHandler(event); });
}

ImageBox.prototype.buildTreeNode = function(config, level, nodeList, parent) {

    var self = this;

    var selectorGroup = document.createElement('div'); 
    selectorGroup.className = "selector-group";

    parent.appendChild(selectorGroup);

    var insets = [];

    for (var i = 0; i < config.length; i++) {
        // Create tab
        var selector = document.createElement('div');
        selector.className = "selector selector-primary";
        // selector.className += (i == 0) ? " active" : "";
        
        selector.addEventListener("click", function(l, idx, event) {
            this.showContent(l, idx);
        }.bind(this, level, i));
        
        // Add to tabs
        selectorGroup.appendChild(selector);

        // Create content
        var contentNode = {};
        contentNode.children = [];
        contentNode.selector = selector;

        var content;
        if (typeof(config[i].elements) !== 'undefined') {
            // Recurse
            content = document.createElement('div');
            this.buildTreeNode(config[i].elements, level+1, contentNode.children, content);
            selector.appendChild(document.createTextNode(config[i].title));
        } else {
            // Create image
            content = document.createElement('img'); 
            content.className = "image-display pixelated";
            content.src = config[i].image;
            wheelzoom(content, options);
            selector.appendChild(document.createTextNode(i+1 + ": " + config[i].title));
            this.selection.length = Math.max(this.selection.length, level+1);

            // Create inset
            var inset = document.createElement('img');
            inset.className = "inset pixelated";
            inset.style.backgroundImage = "url('" + config[i].image + "')";
            inset.style.backgroundRepeat = "no-repeat";
            inset.style.border = "0px solid black";
            inset.style.width  = (options.width / config.length-4) + "px";
            inset.style.height = (options.width / config.length-4) + "px";
            inset.name = config[i].title;
            var canvas = document.createElement("canvas");
            cachedDataUrl = canvas.toDataURL();
            inset.src = cachedDataUrl;
            insets.push(inset);

            content.addEventListener("mousemove", function(content, insets, event) {
                this.mouseMoveHandler(event, content, insets);
            }.bind(this, content, insets));
            content.addEventListener("wheel", function(content, insets, event) {
                this.mouseMoveHandler(event, content, insets);
            }.bind(this, content, insets));

        }
        content.style.display = 'none';

        parent.appendChild(content);
        contentNode.content = content;

        nodeList.push(contentNode);
    }

    if (insets.length > 0) {
        var insetGroup = document.createElement('table');
        insetGroup.className = "insets";
        insetGroup.width = options.width;
        var tr = document.createElement('tr');
        tr.className = "insets";
        insetGroup.appendChild(tr);

        for (var i = 0; i < insets.length; ++i) {
            var auxDiv = document.createElement('td');
            auxDiv.className = "insets";
            auxDiv.style.width = (options.width / insets.length) + "px";
            auxDiv.appendChild(document.createTextNode(insets[i].name));
            auxDiv.appendChild(insets[i]);
            tr.appendChild(auxDiv);
        }

        parent.appendChild(insetGroup);
    }
}

ImageBox.prototype.showContent = function(level, idx) {
    // Hide
    var bgWidth = 0;
    var bgHeight = 0;
    var bgPosX = 0;
    var bgPosY = 0;
    var bgOffset = 0;
    var l = 0;
    var node = {};
    node.children = this.tree;
    while (node.children.length > 0 && node.children.length > this.selection[l]) {
        node = node.children[this.selection[l]];
        node.selector.className = 'selector selector-primary';
        node.content.style.display = 'none';
        if (l == this.selection.length-1) {
            bgWidth =   node.content.bgWidth;
            bgHeight =  node.content.bgHeight;
            bgPosX =    node.content.bgPosX;
            bgPosY =    node.content.bgPosY;
            bgOffset =  node.content.bgOffset;
        }
        l += 1;
    }

    this.selection[level] = Math.max(0, idx);

    // Show
    l = 0;
    node = {};
    node.children = this.tree;
    while (node.children.length > 0) {
        if (this.selection[l] >= node.children.length)
            this.selection[l] = node.children.length - 1;
        node = node.children[this.selection[l]];
        node.selector.className = 'selector selector-primary active';
        node.content.style.display = 'block';
        if (l == this.selection.length-1) {
            node.content.bgWidth = bgWidth;
            node.content.bgHeight = bgHeight;
            node.content.bgPosX = bgPosX;
            node.content.bgPosY = bgPosY;
            node.content.bgOffset = bgOffset;
            node.content.style.backgroundSize = bgWidth+'px '+bgHeight+'px';
            node.content.style.backgroundPosition = bgOffset + bgPosX+'px '+bgPosY+'px';
        }
        l += 1;
    }
}


ImageBox.prototype.keyPressHandler = function(event) {
    if (parseInt(event.charCode) == "0".charCodeAt(0)) {
        var idx = 9;
        this.showContent(this.selection.length-1, idx);
    } else {
        var idx = parseInt(event.charCode) - "1".charCodeAt(0);
        this.showContent(this.selection.length-1, idx);
    }

    // var inc = event.charCode == "+".charCodeAt(0);
    // var dec = event.charCode == "-".charCodeAt(0);
    // if (inc || dec) {
    //     if (inc)
    //         this.insetSize *= 2;
    //     else
    //         this.insetSize /= 2;
    //     for (var i = 0; i < this.elements.length; i++) {
    //         var image = this.insetContainers[i].childNodes[0];
    //         image.style.width = this.insetSize + "px";
    //         image.style.height = this.insetSize + "px";
    //         this.insetContainers[i].style.width = this.insetSize + "px";
    //     }
    // } else {
    //     var idx = parseInt(event.charCode) - "1".charCodeAt(0);
    //     if (idx >= 0 && idx < this.elements.length)
    //         this.selectImage(idx);
    // }
}


ImageBox.prototype.mouseMoveHandler = function(event, image, insets) {
    var rect = image.getBoundingClientRect();
    var xCoord = ((event.clientX - rect.left) - image.bgOffset - image.bgPosX)   / (image.bgWidth  / image.imWidth);
    var yCoord = ((event.clientY - rect.top)  - image.bgPosY)                    / (image.bgHeight / image.imHeight);

    for (var i = 0; i < insets.length; ++i) {
        if (insets[i].name == 'Kernel-KP' || insets[i].name == 'Kernel-WSKP') {
            var scale = 9.5;
            console.log(image.imWidth, image.imHeight);
            insets[i].style.backgroundSize = (1280 * 13 * scale) + "px " + (720 * 13 * scale) + "px";
            var xCoordInt = parseInt(xCoord);
            var yCoordInt = parseInt(yCoord);
            insets[i].style.backgroundPosition = ((insets[i].width/2 - (xCoordInt * 13 + 6.5)*scale)) + "px " + (insets[i].height/2 - (yCoordInt*13+6.5)*scale) + "px";
            

        } else {
            var scale = 2;
            insets[i].style.backgroundSize = (image.imWidth * scale) + "px " + (image.imHeight*scale) + "px";
            insets[i].style.backgroundPosition = ((insets[i].width/2 - xCoord*scale) ) + "px " + (insets[i].height/2 - yCoord*scale) + "px";
        }
    }
}