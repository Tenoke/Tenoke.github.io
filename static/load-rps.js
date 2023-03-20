const fullModel  = tf.loadModel('/static/Full-Model/model.json')
const guid = generateQuickGuid()


currentMoves = []

n_to_move = {
  1:'rock',
  2:'paper',
  3:'scissors'
}

move_to_n = {
  'rock': 1,
  'paper': 2,
  'scissors': 3
}

function getRandomInt(max, starting=1) {
  return (starting + Math.floor(Math.random() * max))
}
//Choose a random first move
let nextComputerMove = getRandomInt(3)
let humanWinCounter = 0
let computerWinCounter = 0
let tieCounter = 0

function to_categorical(n, length=4) {
  let result = Array.from({length}, ()=> 0.)
  result[n] = 1.
  return result
}

function chooseMove(move) {
  console.log('human: ', n_to_move[move])
  var winner = updateCounters(move, nextComputerMove)
  showComputerMove(winner)
  currentMoves.push(to_categorical(move).concat(to_categorical(nextComputerMove)))
  // saveMoves()
  calculateNextComputerMove().then(nextMove=>nextComputerMove=nextMove)
}

function showComputerMove(winner) {
  var element = document.getElementById("ComputerInput");
  switch (winner) {
    case 0:
      element.className = 'btn btn-lg btn-outline-secondary active'
      document.getElementById("ties").textContent = tieCounter
      break;
    case 1:
      element.className = 'btn btn-lg btn-outline-danger active'
      document.getElementById("humanWins").textContent = humanWinCounter
      break;
    case 2:
      element.className = 'btn btn-lg btn-outline-success active'
      document.getElementById("computerWins").textContent = computerWinCounter

  }

  element.innerHTML = `<i class="fa fa-hand-${n_to_move[nextComputerMove]}-o"></i>` 
  element.placeholder = n_to_move[nextComputerMove]
  console.log('pc: ', n_to_move[nextComputerMove])
  console.log('pcwins: ', computerWinCounter, ' humanwins: ', humanWinCounter, ' ties: ', tieCounter)
}


async function calculateNextComputerMove() {
  const model = await fullModel
  let lastMoves = currentMoves.slice(Math.max(currentMoves.length-28, 0), currentMoves.length)
  let data = tf.tensor3d([lastMoves], [1, lastMoves.length, 8])
  let nextHumanMovePrediction = model.predict(data)[0].as1D()
  // we are getting predictions for all moves (because of how we trained the network)
  // but only care for the prediction for the next move - the last 4 numbers
  nextHumanMovePrediction = nextHumanMovePrediction.slice(nextHumanMovePrediction.shape-4,4)
  // next we turn that into a single number from a one-hot encoding
  nextHumanMovePrediction = nextHumanMovePrediction.argMax().dataSync()[0]
  // and turn that into our next move based on what will beat the human
  return moveBasedOn(nextHumanMovePrediction)
}

function moveBasedOn(move) {
  switch(move) {
    case 1:
      return 2
    case 2:
      return 3
    case 3:
      return 1
  }
}

function updateCounters(humanMove, computerMove) {
  switch (humanMove - computerMove) {
    case 0:
      tieCounter++
      return 0
    case  1: 
    case -2:
      humanWinCounter++
      return 1
    default:
      computerWinCounter++
      return 2
  }
}

function saveMoves(){
  if (currentMoves.length % getRandomInt(5, 4) !=0) {
    return
  }
  var xmlhttp = new XMLHttpRequest();   // new HttpRequest instance 
  xmlhttp.open("POST", "/save");
  // xmlhttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  xmlhttp.send(JSON.stringify([guid, currentMoves]));
}

