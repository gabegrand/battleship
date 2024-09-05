import React from "react";
import { usePlayer, usePlayers, useRound } from "@empirica/core/player/classic/react";
import { Button } from "../components/Button";

export function InstructionsWaitingRoom() {
  const player = usePlayer();
  const players = usePlayers();
  const round = useRound();

  return (
    <div style={{display:"flex", flexDirection:"column", alignItems:"center", gap:"10px"}}>
      <p style={{fontSize:"1.5vw"}}>You have finished the tutorial: well done!</p>
      <p style={{fontSize:"1.5vw"}}>Click the button below, and you will progress to the rest of the game when the other player finishes it.</p>
      <p style={{fontSize:"1.5vw"}}>As a reminder, you will now complete 2 training games, followed by 6 test games.</p>
      <p style={{fontSize:"1.5vw"}}>You will switch roles every game, so don't worry if you find one role more boring than the other, you won't be stuck with it.</p>
      <p style={{fontSize:"1.5vw"}}>Good luck, and we hope you have fun!</p>
      <div style={{margin:"10px"}}>
      <Button handleClick={() => player.stage.set("submit", true)}>
        Next Game
      </Button>
      </div>
    </div>
  );
}