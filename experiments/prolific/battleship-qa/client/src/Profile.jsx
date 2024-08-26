import {
  usePlayer,
  useRound,
  useStage,
  useGame
} from "@empirica/core/player/classic/react";
import React from "react";
import { Avatar } from "./components/Avatar";
import { Timer } from "./components/Timer";
import { StageRectangles } from "./components/StageRectangles.jsx";
import Board, { hasGameEnded, hasShipSunk } from "../../client/src/stages/Board";
import { StuckButton } from "./components/StuckButton.jsx";


export function Profile() {
  const player = usePlayer();
  const round = useRound();
  const stage = useStage();
  const game = useGame();

  function handleStuck() {
    var confirmation = window.confirm("WARNING: clicking 'OK' will end the game *for both you AND your partner*. Please only click this in the case of an emergency, if it truly looks like the game is stuck and has been for more than 10 minutes. If the game ends this way, the experimenters may reach out, and we will compensate you for your time.");
    if (confirmation) {
      game.set("stuck",true);
      player.stage.set("submit",true);
    }
  }

  function getBonusPercent() {

    const penalty = 15;

    const moves = round.get("moves");
    const tiles = round.get("trueTiles");
    var misses = 0;
    if (tiles != undefined && moves != undefined) {
      var moveCoords = moves.map((move) => ([move[0].charCodeAt()-"A".charCodeAt(),move[1]-1]));
      var missArray = moveCoords.map((coord) => (tiles[coord[0]][coord[1]] == 0 ? 1 : 0));
      var misses = missArray.reduce((partialSum, a) => partialSum + a, 0);
    }

    const bonus_pct = Math.max(100-penalty*misses,20).toString();

    round.set("bonus", bonus_pct);
    return bonus_pct+"%";
  }

  function getSignal() {
    switch (stage.get("name")) {
      case "shared_stage":
        return 0;
      case "shared_stage_2":
        return 1;
      case "shared_stage_3":
        return 2;
      case "instructions":
        switch (player.stage.get("textStage")) {
          case 6:
            return 2;
          case 9:
            return 1;
          default:
            return 0;
        }
    }
  }

  return (
    <div className="min-w-xl md:min-w-2xl mt-2 m-x-auto px-3 py-2 text-gray-500 rounded-md grid grid-cols-3 items-center border-.5" style={{width:"10vw"}}>
      <div className="leading-tight ml-1" style={{display:"flex",flexDirection:"column"}}>
        <div className="text-gray-600 font-semibold" style={{fontSize:"1.4vw"}}>
          Round 
        </div>
        <div style={{fontSize:"1.7vw", paddingLeft:"0.6vw"}}>{game.get("elapsedRounds")+1}/{game.get("totalRounds")}</div>
      </div>

      <div style={{display:"flex", flexDirection:"column", alignItems:"center"}}>
      <Timer />
      <StageRectangles 
          signal={getSignal()}/>
      <StuckButton handleClick={handleStuck}>End Experiment</StuckButton>
      </div>

      <div className="flex space-x-3 items-center justify-end">
        <div className="flex flex-col items-center">
          {stage.get("name") == "instructions"
          ? <div></div>
          : (<div style={{display:"flex", flexDirection:"column", alignItems:"center"}}><div className="text-gray-600 font-semibold" style={{fontSize:"1.4vw"}}>
          Performance
        </div>
        <div style={{fontSize:"1.6vw"}}>
          {getBonusPercent()}
        </div></div>)

          }
          
        </div>
      </div>

    </div>
  );
}
