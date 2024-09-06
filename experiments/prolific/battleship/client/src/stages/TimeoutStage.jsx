import React from "react";
import { usePlayer, usePlayers, useRound, useGame } from "@empirica/core/player/classic/react";
import { Button } from "../components/Button";

export function TimeoutStage() {
  const player = usePlayer();
  const game = useGame();

  function handleAttentionCheck() {
    player.set("timeoutGuiltySticky",false);
    player.set("timeoutGuilty",false);
    player.stage.set("timedOut",false);
    player.stage.set("submit", true);
  }

  return (
    <div style={{display:"flex", flexDirection:"column", alignItems:"center"}}>
      <p style={{fontSize:"2vw"}}> It appears {player.get("timeoutGuiltySticky") ? "You have" : "Your partner has"} been unresponsive for an extended period. Please confirm you are still actively participating in this study.</p>
      <p style={{fontSize:"2vw"}}>If you and your partner do not both respond within the time limit, the game will automatically end, and {player.get("timeoutGuiltySticky") ? "your submission will be rejected." : "you will be compensated for the time spent in the game so far."}</p>
      <p style={{fontSize:"1.5vw"}}>Press the button below to continue the game.</p>
      <div style={{margin:"10px"}}>
      <Button handleClick={() => handleAttentionCheck()}>
        I'm here!
      </Button>
      </div>
    </div>
  );
}