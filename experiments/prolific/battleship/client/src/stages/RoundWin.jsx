import React from "react";
import { usePlayer, usePlayers, useRound } from "@empirica/core/player/classic/react";
import { Button } from "../components/Button";

export function RoundWin() {
  const player = usePlayer();
  const players = usePlayers();
  const round = useRound();

  return (
    <div style={{display:"flex", flexDirection:"column", alignItems:"center"}}>
      <p style={{fontSize:"18px"}}>You found all the ships in <b>{round.get("score")}</b> moves! Click the button below to continue to the next round.</p>
      <div style={{margin:"10px"}}>
      <Button handleClick={() => player.stage.set("submit", true)}>
        Next Game
      </Button>
      </div>
    </div>
  );
}