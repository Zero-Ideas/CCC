// Assuming you're working in Adobe Animate (HTML5 Canvas)
var deathScreen = this.diedScren
var deathScoreText = this.score2
var root = this
var winPrompt = root.winPrompt
var replayButton = 
deathScreen.visible = false
deathScoreText.visible = false
winPrompt.visible = false



stage.enableMouseOver(); // Enable mouse over effect
let scoreText = this.skore



scoreText.text = 5;
// Assuming `this.b1`, `this.b2`, etc., are valid display objects in the stage
let balls = [this.b1, this.b2, this.b3, this.b4, this.b5];
let deadBalls = []


console.log('test');
console.log(balls);
var score = 0
var hp = 5;

var fallspeed = 5

// Function to iterate thSrough the balls array

function drop() {
  // Iterate through the balls array
	
	for (let i = 0; i < balls.length; i++) {
		// Access each ball in the array and perform operations
		//console.log(balls[i]);  // Example operation: logging each ball
		if (balls[i].y >= 550) {
			if ( deadBalls.includes(balls[i]) == false) {
				deadBalls.push(balls[i])
				hp--
				console.log(hp)
				if (hp == 0) {
					scoreText.text = "you died"
					deathScreen.visible = true
					
					deathScoreText.text = score
					deathScoreText.visible = true
					console.log(deathScoreText.visible)
					break
				}
			}
			continue
		}
		balls[i].y += fallspeed
		scoreText.text = score
	}
}




for (let i = 0; i < balls.length; i++) {
    // Access each ball in the array and perform operations
	console.log(balls[i]);  // Example operation: logging each ball
	balls[i].addEventListener("click",function(){
		score++
		balls[i].y = 0;
		console.log(score)
		
		fallspeed += 0.05
	});
	
}


var startButton = this.start

// Add a click event listener to the start button
startButton.addEventListener("click", fl_MouseClickHandler);

function fl_MouseClickHandler(event) {
  // Custom code that runs on mouse click
	console.log("Mouse clicked");
	startButton.visible = false
  // Call drop function when clicked (if needed)
  
	createjs.Ticker.setInterval(25); // Set ticker interval to 100ms
	createjs.Ticker.addEventListener("tick", drop); // Add a listener for the tick event
}	

function loopDrop(event) {
  // Call the drop function
  drop();
}