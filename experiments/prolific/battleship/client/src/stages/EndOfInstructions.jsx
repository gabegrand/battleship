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
      <p style={{fontSize:"1.25vw"}}>You successfully completed the tutorial!</p>
      <p style={{ fontSize: "1.25vw" }}>You will now proceed to the study, which is designed to last approximately <b>1 hour</b>. Please remember:</p>
      <ul style={{fontSize:"1.25vw"}}>
        <li>Every hit increases your bonus! You have the opportunity to earn <b>$12.00 bonus</b> or more based on your performance.</li>
        <li>When playing as Captain, use your questions to help inform your moves. This is a key part of the study, so poor quality / low-effort questions will not receive any bonus.</li>
        <li>Every miss reduces your bonus, so take your time to think through your actions.</li>
        <li>You are playing with another real person, so please be patient and responsive.</li>
        <li>Do not navigate away from the study or attempt to multi-task. If you are unresponsive for an extended period, the study will end and your submission will be rejected.</li>
      </ul>
      <div style={{margin:"10px"}}>
      <Button handleClick={() => handleEnd()}>
        Continue
      </Button>
      </div>
    </div>
  );
}