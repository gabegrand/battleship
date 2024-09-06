import React from "react";
import { usePlayer, usePlayers, useRound } from "@empirica/core/player/classic/react";
import { Button } from "../components/Button";

export function EndOfInstructions() {
  const player = usePlayer();
  const players = usePlayers();
  const round = useRound();

  
  function handleEnd(){
    player.stage.set("timedOut",false);
    player.stage.set("submit",true);
  }

  return (
    <div style={{display:"flex", flexDirection:"column", alignItems:"center"}}>
      <p style={{fontSize:"1.25vw"}}>You successfully completed the tutorial.</p>
      <p style={{fontSize:"1.25vw"}}> <b> Are you prepared to spend the next hour working collaboratively with your partner to play through the game? </b></p>
      <p style={{fontSize:"1.25vw"}}>If so, click the button below and continue to the first game.</p>
      <div style={{margin:"10px"}}>
      <Button handleClick={() => handleEnd()}>
        Yes
      </Button>
      </div>
    </div>
  );
}