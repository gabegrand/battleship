import { ClassicListenersCollector } from "@empirica/core/admin/classic";
export const Empirica = new ClassicListenersCollector();
import Board, { hasGameEnded, hasShipSunk } from "../../client/src/stages/Board";
import { gameBoards, testBoards } from "../../utils/authoredBoards";
import { SendMessage } from "../../utils/SendMessage";

function handleTimeOut(stageName, round) {
  switch (stageName) {
    case "shared_stage":
      SendMessage("(question timed out)","question",round,"timeout");
      break;
    case "shared_stage_2":
      SendMessage("(answer timed out)","answer",round,"timeout");
      round.set("spotterTimedOut",true);
      break;
    case "shared_stage_3":
      SendMessage("(firing timed out)","move",round,"timeout");
      round.set("firingTimedOut",true);
      break;
  }
}

Empirica.onGameStart(({ game }) => {
  const treatment = game.get("treatment");
  ///HYPERPARAMETERS
  game.set("testRounds",0);
  game.set("totalRounds",treatment.realNumber);
  game.set("roundDuration",treatment.maxTime);
  game.set("timeoutDuration",treatment.timeoutDuration);
  game.set("totalQuestions",treatment.totalQuestions);
  game.set("categoricalAnswers",treatment.categorical);
  game.set("spotterRatesQuestions",treatment.spotterRatesQuestions);
  game.set("questionEveryTime",treatment.questionEveryTime);
  game.set("uniformStart",treatment.uniformStart);
  game.set("startingBonus",treatment.startingBonus);
  game.set("missPenalty",treatment.missPenalty);
  game.set("skipTutorial",treatment.skipTutorial);
  game.set("bonusFloor",0);
  game.set("timeoutEnd",false);
  game.set("playersTimedOut",false);

  game.set("compcode", treatment.compcode);
  game.set("compcodeStuck", treatment.compcodeStuck);
  game.set("compcodeTimeoutGuilty", treatment.compcodeTimeoutGuilty);
  game.set("compcodeTimeoutInnocent", treatment.compcodeTimeoutInnocent);
  ////----------------

  if (game.get("questionEveryTime")) {
    game.set("totalQuestions",1000);
  }

  game.set("elapsedRounds",0);
  game.set("instructionsJustFinished",false);
  game.set("stuck",false);
  game.set("bonuses",[]);
  game.set("timeoutGuiltAssigned",false);

  const players = game.players;
  for (const player of players) {
    player.set("stuck",false);
    player.set("timeoutGuilty",false);
    player.set("timeoutGuiltySticky",false);
    player.set("timeoutGameEnd",false);
  }

  game.set("gameBoards",gameBoards);
  game.set("testBoards",testBoards);

  const round = game.addRound({
    name: `Round`,
  });
  round.addStage({ name: "instructions", duration: 300 });

});

Empirica.onRoundStart(({ round }) => {

  var game = round.currentGame;
  var elapsedRounds = game.get("elapsedRounds");
  var testRounds = game.get("testRounds");

  round.set("questionsRemaining", game.get("totalQuestions"));

  const players = game.players;

  var even_round = (elapsedRounds % 2) == 0;

  const spotter = even_round ? players[0] : players[1];
  const captain = even_round ? players[1] : players[0];

  spotter.round.set("role","spotter");
  spotter.set("name", "Spotter");

  captain.round.set("role","captain");
  captain.set("name", "Captain");

  var boardListName = elapsedRounds >= testRounds
                      ? "gameBoards" 
                      : "testBoards";

  var boardList = game.get(boardListName);
 
  var shipSelected = boardList[elapsedRounds];

  round.set("board_id", shipSelected[0]);
  round.set("trueTiles", shipSelected[1]);

  var occTiles = shipSelected[2]; 
  if (game.get("uniformStart")) {
    var boardLen = shipSelected[1][0].length;
    occTiles = Array.from({ length: boardLen}, () => Array(boardLen).fill(-1));
  } 

  round.set("occTiles", occTiles);
  round.set("ships", shipSelected[3]);

  round.set("score",0);

  var board_real = new Board(shipSelected[1][0].length, shipSelected[1], shipSelected[3]);
  round.set("shipsSunk", round.get("ships").map((ship) => [ship, [false,board_real.getShipLength(ship)]]));

  round.set("moves",[]);
  round.set("messages", []);
  round.set("spotterRatings",[]);
  round.set("gameOver",false);

  round.set("skippedToFiring",false);
  round.set("spotterTimedOut",false);
  round.set("firingTimedOut",false);

});


Empirica.onStageStart(({ stage }) => {
  stage.set("questionAsked",false);
  stage.set("questionRated",false);
  stage.set("answered",false);
  stage.set("timeoutHandled",false);

  var players = stage.currentGame.players;

  players[0].stage.set("timedOut",true);
  players[1].stage.set("timedOut",true);

});

Empirica.onStageEnded(({ stage }) => {
  const game = stage.currentGame;
  const round = stage.round;
  const players = game.players;

  //console.log(round.get("messages"));

  var filteredMessages = [];
  let previousType = "";
  for (const message of round.get("messages")) {
    if (message.type == previousType) {
      continue;
    }
    filteredMessages.push(message);
    previousType = message.type;
  }

  //round.set("messages",filteredMessages);

  var timeOutArray = [players[0].stage.get("timedOut"),players[1].stage.get("timedOut")];

  players[0].set("timeoutGuilty", players[0].stage.get("timedOut"));
  players[1].set("timeoutGuilty", players[1].stage.get("timedOut"));

  var playersTimedOut = timeOutArray.some((bool) => bool == true);

  if (playersTimedOut && !game.get("timeoutGuiltAssigned")) {
    game.set("playersTimedOut",playersTimedOut);
    players[0].set("timeoutGuiltySticky",players[0].get("timeoutGuilty"));
    players[1].set("timeoutGuiltySticky",players[1].get("timeoutGuilty"));
    game.set("timeoutGuiltAssigned",true);

    if (!stage.get("timeoutHandled")) {
    handleTimeOut(stage.get("name"),round);
    stage.set("timeoutHandled",true);
    }
  }

  var timeoutEnd = false;
  if (stage.get("name") == "timeout") {
    var timeoutEnd = [players[0].get("timeoutGuilty"),players[1].get("timeoutGuilty")].some((bool) => bool == true);
    game.set("timeoutEnd",timeoutEnd);
    //console.log("timeoutEnd",[players[0].get("timeoutGuilty"),players[1].get("timeoutGuilty")],timeoutEnd);
    if (timeoutEnd) {
      players[0].set("timeoutGameEnd",true);
      players[1].set("timeoutGameEnd",true);
      return 0;
    } else {
      game.set("timeoutGuiltAssigned",false);
    }
  }

  if (!game.get("stuck") && !timeoutEnd) {

    if ((stage.get("name") == "shared_stage" || stage.get("name") == "shared_stage_2" || stage.get("name") == "shared_stage_3" || stage.get("name") == "timeout")) {
      var tiles_uncovered = round.get("occTiles");
      var tiles_true = round.get("trueTiles");
      var round_ships = round.get("ships");
    
      var board_uncovered = new Board(tiles_uncovered[0].length, tiles_uncovered, round_ships);
      var board_true = new Board(tiles_uncovered[0].length, tiles_true, round_ships);
    
      var ships_sunk = round_ships.map((ship) => [ship, [hasShipSunk(board_uncovered, board_true, ship), board_true.getShipLength(ship)]]);

      round.set("shipsSunk",ships_sunk);

      switch (stage.get("name")) {
        case "timeout":
          game.set("playersTimedOut",false);
          round.set("spotterTimedOut",false);
          round.set("firingTimedOut",false);
          round.addStage({ name: "shared_stage", duration: game.get("roundDuration") });
          break;
        case "shared_stage":
          if (round.get("gameOver")) {
            break;
          } else {
            round.addStage({ name: "shared_stage_2", duration: game.get("roundDuration") });
          }
          break;
        case "shared_stage_2":
          round.addStage({ name: "shared_stage_3", duration: game.get("roundDuration") });
            break;
        case "shared_stage_3":
            round.set("questionSkipped",false);
            round.set("question", undefined);
            round.set("answer", undefined);
            round.set("score", round.get("score")+1); //increases move count by one even if they don't fire
            round.set("skippedToFiring",false);
            if (game.get("playersTimedOut")) {
              round.addStage({ name: "timeout", duration: game.get("timeoutDuration")});
            } else {
              round.addStage({ name: "shared_stage", duration: game.get("roundDuration") });
            }
            
          break;
      }
    }
  }
    if (stage.get("name") == "instructions") {
      game.set("instructionsJustFinished",true);
    }

  else {}

});

Empirica.onRoundEnded(({ round }) => {
  curr_game = round.currentGame;

  if (!curr_game.get("stuck") && !curr_game.get("timeoutEnd")) {
    var prevRounds = curr_game.get("elapsedRounds");
    curr_game.set("bonuses",[...curr_game.get("bonuses"), round.get("bonus")])

    if (curr_game.get("instructionsJustFinished")) {
      var nextRoundNumber = prevRounds;
    } else {
      var nextRoundNumber = prevRounds+1;
    }
    curr_game.set("instructionsJustFinished",false);
    curr_game.set("elapsedRounds", nextRoundNumber);

    if (nextRoundNumber < curr_game.get("totalRounds")) {
      const new_round = curr_game.addRound({
        name: `Round`,
      });
      if (nextRoundNumber == 0) {
        new_round.addStage({ name: "end_of_instructions", duration: curr_game.get("roundDuration") });
      }
      if (nextRoundNumber == curr_game.get("testRounds") && curr_game.get("testRounds") != 0) {
        new_round.addStage({ name: "end_of_tests", duration: curr_game.get("roundDuration") });
      }
      new_round.addStage({ name: "shared_stage", duration: curr_game.get("roundDuration") });
    }
  }
  else {}

});

Empirica.onGameEnded(({ game }) => {
    const players = game.players;
    for (const player of players) {
      player.set("compcodeTimeoutGuilty",game.get("compcodeTimeoutGuilty"));
      player.set("compcodeTimeoutInnocent",game.get("compcodeTimeoutInnocent"));
      player.set("compcodeStuck",game.get("compcodeStuck"));
      player.set("compcode",game.get("compcode"));
    }
});

Empirica.on("game", "stuck", (ctx, { game, stuck }) => {
  if (stuck) {
    const players = game.players;
    for (const player of players) {
      player.set("stuck",true);
      player.stage.set("submit",true);
    }
  }
});
