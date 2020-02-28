var data = [];
var w = [1,1,1];

var m = 1;
var b = 0;

function testMathJs(){
	m = math.matrix([[1,2],[3,4]]);
	print(math.multiply(m, m));
	print(math.log(m));
	print(math.add(1,m));
	print(math.sum(m));
 	
 	v = math.matrix([1,2,3,4]);
 	print(math.multiply(v,v));
 	print(math.subtract(v,2))

}

function randomInitWeight(){
	w = [Math.random(), Math.random(), Math.random()]
	print(w);
	return w;
}

function sigmoid(zs){
	res = zs.map(z => 1 / (1 + exp(-1 * z)));
	return res;
}

function hx(w, data){
	var xs = data.map(v=>v.x);
	var ys = data.map(v=>v.y);
	var zs = [];
	for (var i=0;i<data.length;i++){
		z = w[0]+(xs[i])*w[1]+(ys[i])*w[2];
		zs.push(z);
	}
	return sigmoid(zs);
}

function cost(w, data){
	var pred = hx(w,data);
	pred = math.matrix(pred);

	
	var Z = data.map(v=>v.z);
	
	
	// print("g1: ")
	// print(pred);
	// print(math.subtract(1, pred));
	// print(math.log(math.subtract(1, pred)));
	// print(math.subtract(1, Z));
	// print(math.multiply(math.subtract(1, Z), math.log(math.subtract(1, pred))));

	// print("g2: ")
	// print(Z);
	// print(pred);
	// print(math.log(pred));
	// print(math.multiply(Z, math.log(pred)));

	var s1 = math.multiply(Z, math.log(pred));
	var s2 = math.multiply(math.subtract(1, Z), math.log(math.subtract(1, pred)));

	return -1 * (s1+s2);
}

function grad(w, data){

	var m = data.length;
	var pred = hx(w,data);
	var Z = data.map(v=>v.z);
	var x1 = data.map(v=>v.x);
	var x2 = data.map(v=>v.y);

	var g0 = math.sum(math.multiply(1/m, math.subtract(pred, Z)));
	// print(g0);
	var g1 = math.multiply(1/m, math.multiply(x1, math.subtract(pred, Z)));
	// print(g1);
	var g2 = math.multiply(1/m, math.multiply(x2, math.subtract(pred, Z)))
	// print(g2);

	return [g0,g1,g2];
}

function descent(newW, prevW, lr, data){
	var m = data.length;
	if (m == 0) {
		return;
	}
	// print(prevW);
	// print(cost(prevW, data));
	j=0;
	while (true) {
		prevW = newW;
		newW[0] = newW[0] - lr * grad(prevW, data)[0];
		newW[1] = newW[1] - lr * grad(prevW, data)[1];
		newW[2] = newW[2] - lr * grad(prevW, data)[2];
		// print(newW);
		// print("prevWCost: " + cost(prevW, data));
		// print("newWCost: " + cost(newW, data));

		w = newW;
		drawLogisticRegressionLine(w);
		if (j>100){
			return newW;
		}
		j+=1;
	}
}

function testGrad(){
	grad([1,1,1], initDataPoints());
}

function testDescent(){
	w = [1,2,3];
	descent(w,w,0.3, initDataPoints());
}

function testCost(){
	var w = [1,2,3];
	var data = initDataPoints();
	cost(w, data);
}

function initDataPoints(){
	return [];
	return [createVector(0.3,0.2,1), createVector(0.8,0.2,1)]; //vert 
	return [createVector(0.3,0.2,1), createVector(0.34,0.8,1)]; //hori
	return [createVector(0.3,0.2,1), createVector(0.4,0.7,0), createVector(0.6,0.9,1)];
}

function testHx(){
	hx([2,2,2],initDataPoints());

}

function setup() {
  var canvas = createCanvas(500, 500);
  canvas.mousePressed(onMousePressed);
  data = initDataPoints();
  w = randomInitWeight();
  print(data);
  print(w);


	 

}

function gradientDescent(){
	var learning_rate = 0.5;
	var m_grad = 0;
	var b_grad = 0;

	var iteration = 100;
	
	for(var i=0;i<data.length;i++){
		var x=data[i].x;
		var y=data[i].y;
		var cost = m*x+b - y;
		m_grad += cost * x;
		b_grad += cost ;
	}
	m_grad = learning_rate / data.length * m_grad;
	b_grad = learning_rate / data.length * b_grad;
	// print(m_grad);
	m = m - m_grad;
	b = b - b_grad;

	drawLinearRegressionLine();
}

function linearRegression(){
	//(sum(x-xmean)*sum(y-ymean))/sum((x-xmean)^2)
	var xsum=0;
	var ysum=0;
	for(var i=0;i<data.length;i++){
		xsum+=data[i].x;
		ysum+=data[i].y;
	}
	var xmean=xsum/data.length;
	var ymean=ysum/data.length;
	var num=0;
	var dem=0;
	for(var i=0;i<data.length;i++){
		x=data[i].x;
		y=data[i].y;
		num+=(x-xmean)*(y-ymean);
		dem+=(x-xmean)*(x-xmean);
	}
	
	m=num/dem;

	b=ymean-m*xmean;

	drawLinearRegressionLine();
}

function drawLogisticRegressionLine(w){
	// y = -(w[0]-w[1]*x)*(1/w[2])
	var x1 = 0;
	var y1 = (-w[0]-w[1]*x1)*(1/w[2]);
	var x2 = 1;
	var y2 = (-w[0]-w[1]*x2)*(1/w[2]);

	var p1 = decodeDataPoint(x1,y1);
	var p2 = decodeDataPoint(x2,y2);

	stroke(255, 204, 0);
	line(p1.x,p1.y,p2.x,p2.y);
}

function drawLinearRegressionLine(){
	var x1=0;
	var x2=1;
	var y1=m*x1+b;
	var y2=m*x2+b;

	// x1=map(x1,0,1,0,width);
	// y1=map(y1,0,1,height,0);
	// x2=map(x2,0,1,0,width);
	// y2=map(y2,0,1,height,0);

	var p1 = decodeDataPoint(x1,y1);
	var p2 = decodeDataPoint(x2,y2);

	// stroke('rgba(50,255,50, 0.6)');
	stroke(255, 204, 0);
	line(p1.x,p1.y,p2.x,p2.y);
}

function decodeDataPoint(dataX, dataY){
	var canvasX = map(dataX,0,1,0,width);
	var canvasY = map(dataY,0,1,height,0);
	return {x:canvasX, y:canvasY};
}

function encodeCanvasPoint(canvasX, canvasY){
	var dataX = map(canvasX, 0, width, 0, 1);
	var dataY = map(canvasY,0,height,1,0);
	return {x:dataX, y:dataY};
}

function onMousePressed(){
 	var z;
  	if(mouseIsPressed){
		if(mouseButton === LEFT){
			z = 1;
		}else if(mouseButton === RIGHT){
			z = 0;
		}else{
			return
		}
  	}
  	print(mouseX, mouseY)
  	pushData(mouseX,mouseY,z)
  	
}

function pushData(mX,mY,z){
	// var x = map(mX, 0, width, 0, 1);
	// var y = map(mY, 0, height, 1, 0);
	dataPoint = encodeCanvasPoint(mX,mY);
	// print(x, y)
	var point = createVector(dataPoint.x,dataPoint.y,z);
 	data.push(point);
}

function draw() {

  background('#222222');

  fill(color('magenta'));
  stroke('#222222');
  text("Right click", width - 80, height - 30)

  fill(255);
  stroke('#222222');
  text("Left click", width - 140,height - 30)

  for(var i=0;i<data.length;i++){
    // var x = map(data[i].x,0,1,0,width);
    var canvasPoint = decodeDataPoint(data[i].x, data[i].y);
    // print(x);
    // var y = map(data[i].y,0,1,height,0);
    // print(y);
    var z = data[i].z;
    // print(z);
    var c;
    if(z){
    	// left click, z = 1
    	c = 255;
	}else{
		// right click, z = 0
		c = color('magenta')
	}
	fill(c);
   	stroke(c);
    
    ellipse(canvasPoint.x,canvasPoint.y,8,8);

  }
  if (data.length>1) {
  	descent(w,w,0.2,data);
  	// linearRegression()
  	// gradientDescent();  	
  }
}