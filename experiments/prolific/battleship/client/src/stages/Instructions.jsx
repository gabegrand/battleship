import React, { useState } from "react";
import { usePlayer, useStage, useGame, useRound, useStageTimer } from "@empirica/core/player/classic/react";
import { IntroStage1 } from "./IntroStage1.jsx";

export function Instructions() {

  const player = usePlayer();
  const round = useRound();
  const stage = useStage();
  const game = useGame();
  const timer = useStageTimer();

  if (player.stage.get("textStage") == undefined) {
    player.stage.set("textStage",1);
  }

  player.stage.set("ships",[1,2]);
  player.stage.set("questionsRemaining",2);

  if (player.stage.get("messages") == undefined) {
    player.stage.set("messages",[]);
  }

  player.stage.set("shipStatus",[[1,[false,3]],[2,[true,4]]]);

  if (player.stage.get("occTiles") == undefined || player.stage.get("textStage") == 8) {
    player.stage.set("occTiles",[
        [ 0, 2, -1, -1, -1],
        [-1, 2, -1, -1, -1],
        [-1,  2,  -1,  1, -1],
        [-1,  2, -1, -1, -1],
        [-1,  0,  0, -1, -1]]);
    }

  player.stage.set("trueTiles",  
      [[0,2,0,0,0],
      [0,2,0,1,0],
      [0,2,0,1,0],
      [0,2,0,1,0],
      [0,0,0,0,0]]);

  if (game.get("skipTutorial")) {
    player.stage.set("timedOut",false);
    player.stage.set("submit",true);
  }

    return (<IntroStage1></IntroStage1>);
}