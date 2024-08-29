import React from "react";
import { usePlayer, usePlayers, useRound } from "@empirica/core/player/classic/react";
import { Button } from "../components/Button";

export function EndOfTests() {
  const player = usePlayer();
  const players = usePlayers();
  const round = useRound();

  return (
    <div style={{display:"flex", flexDirection:"column", alignItems:"center"}}>
      <p style={{fontSize:"18px"}}>You successfully completed all of the training rounds. Your next game will be the first test game -- click the button below when you're ready.</p>
      <div style={{margin:"10px"}}>
      <Button handleClick={() => player.stage.set("submit", true)}>
        Continue
      </Button>
      </div>
    </div>
  );
}