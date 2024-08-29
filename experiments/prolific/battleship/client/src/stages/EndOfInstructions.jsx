import React from "react";
import { usePlayer, usePlayers, useRound } from "@empirica/core/player/classic/react";
import { Button } from "../components/Button";

export function EndOfInstructions() {
  const player = usePlayer();
  const players = usePlayers();
  const round = useRound();

  return (
    <div style={{display:"flex", flexDirection:"column", alignItems:"center"}}>
      <p style={{fontSize:"18px"}}>You successfully completed the tutorial. Press the button below to continue to the training rounds!</p>
      <div style={{margin:"10px"}}>
      <Button handleClick={() => player.stage.set("submit", true)}>
        Continue
      </Button>
      </div>
    </div>
  );
}